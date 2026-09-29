import { afterEach, describe, expect, test, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { getOAuthProviders } from "@oh-my-pi/pi-ai/registry/oauth";
import { getEnvApiKey } from "@oh-my-pi/pi-ai/stream";
import { providerEntry } from "@oh-my-pi/pi-catalog/compat/providers";
import { resolveProviderModels } from "@oh-my-pi/pi-catalog/model-manager";
import { getBundledModels } from "@oh-my-pi/pi-catalog/models";
import { DEFAULT_MODEL_PER_PROVIDER, PROVIDER_DESCRIPTORS } from "@oh-my-pi/pi-catalog/provider-models/descriptors";
import {
	isRouterbaseChatModality,
	routerbaseModelManagerOptions,
} from "@oh-my-pi/pi-catalog/provider-models/openai-compat";
import type { FetchImpl, ResolvedOpenAICompat } from "@oh-my-pi/pi-catalog/types";

const DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash";
const ORIGINAL_ROUTERBASE_API_KEY = Bun.env.ROUTERBASE_API_KEY;

afterEach(() => {
	if (ORIGINAL_ROUTERBASE_API_KEY === undefined) {
		delete Bun.env.ROUTERBASE_API_KEY;
	} else {
		Bun.env.ROUTERBASE_API_KEY = ORIGINAL_ROUTERBASE_API_KEY;
	}
	vi.restoreAllMocks();
});

function modelsResponse(data: unknown[]): FetchImpl {
	return vi.fn(
		async () =>
			new Response(JSON.stringify({ object: "list", data }), {
				status: 200,
				headers: { "Content-Type": "application/json" },
			}),
	) as unknown as FetchImpl;
}

describe("RouterBase provider support", () => {
	test("registers the provider, its ROUTERBASE_API_KEY fallback, and a login flow", () => {
		Bun.env.ROUTERBASE_API_KEY = "sk-rb-test";
		expect(getEnvApiKey("routerbase")).toBe("sk-rb-test");
		delete Bun.env.ROUTERBASE_API_KEY;
		expect(getEnvApiKey("routerbase")).toBeUndefined();

		const descriptor = PROVIDER_DESCRIPTORS.find(item => item.providerId === "routerbase");
		expect(descriptor?.defaultModel).toBe(DEFAULT_MODEL);
		expect(descriptor?.catalogDiscovery?.envVars).toEqual(["ROUTERBASE_API_KEY"]);
		expect(DEFAULT_MODEL_PER_PROVIDER.routerbase).toBe(DEFAULT_MODEL);
		// The default must resolve from the bundle before the first discovery.
		expect(getBundledModels("routerbase").map(model => model.id)).toContain(DEFAULT_MODEL);

		expect(getOAuthProviders().find(item => item.id === "routerbase")?.name).toBe("RouterBase");
	});

	test("bundles what RouterBase charges and speaks its request dialect", () => {
		const byId = new Map(getBundledModels("routerbase").map(model => [model.id, model]));

		// Charged per-1M rates, not the list price /v1/models reports.
		const flash = byId.get(DEFAULT_MODEL);
		expect(flash?.cost).toEqual({ input: 0.3, output: 1.2, cacheRead: 0.006, cacheWrite: 0 });
		expect(flash?.contextWindow).toBe(1_048_576);
		expect(flash?.maxTokens).toBe(393_216);
		expect(flash?.input).toEqual(["text", "image"]);
		// A row without a cache tariff bills cached input at the input rate.
		expect(byId.get("qwen/qwen3-coder-next")?.cost).toEqual({
			input: 0.17,
			output: 1.275,
			cacheRead: 0.17,
			cacheWrite: 0,
		});
		expect(byId.get("deepseek/deepseek-v4-pro")?.input).toEqual(["text"]);

		// Same ids exist on other hosts with other limits; the seed must win.
		expect(providerEntry("routerbase")?.skipCrossProviderReferenceFills).toBe(true);

		for (const model of byId.values()) {
			const compat = model.compat as ResolvedOpenAICompat;
			// RouterBase types `max_tokens` only and has no `developer` role.
			expect(compat.maxTokensField).toBe("max_tokens");
			expect(compat.supportsDeveloperRole).toBe(false);
			expect(compat.reasoningContentField).toBe("reasoning_content");
		}
	});

	test("discovery drops non-chat modalities and keeps the seed tariff over the listed price", async () => {
		const fetchMock = modelsResponse([
			{
				id: DEFAULT_MODEL,
				object: "model",
				owned_by: "DeepSeek",
				modality: "chat",
				context_length: 524_288,
				max_completion_tokens: 65_536,
				input_modalities: ["text", "image"],
				supported_parameters: ["tools", "tool_choice", "reasoning"],
				pricing: { prompt: "9.99", completion: "9.99", unit: "1M tokens" },
			},
			{ id: "bytedance/seedream-4.5", modality: "image" },
			{ id: "openai/sora-2-t2v", modality: "video" },
			{ id: "openai/text-embedding-3-large", modality: "embedding" },
			{ id: "vendor/some-tts", modality: "audio" },
		]);

		const models = await routerbaseModelManagerOptions({
			apiKey: "sk-rb-key",
			fetch: fetchMock,
		}).fetchDynamicModels?.();

		expect(models?.map(model => model.id)).toEqual([DEFAULT_MODEL]);
		expect(fetchMock).toHaveBeenCalledWith(
			"https://routerbase.com/v1/models",
			expect.objectContaining({
				method: "GET",
				headers: expect.objectContaining({ Authorization: "Bearer sk-rb-key" }),
			}),
		);

		const discovered = models?.[0];
		// Live limits win over the seed's.
		expect(discovered?.contextWindow).toBe(524_288);
		expect(discovered?.maxTokens).toBe(65_536);
		// `pricing` is list price; the bundled charged rate stays.
		expect(discovered?.cost).toEqual({ input: 0.3, output: 1.2, cacheRead: 0.006, cacheWrite: 0 });
	});

	test("an id with no seed row takes its limits and capabilities from the /v1/models row", async () => {
		const fetchMock = modelsResponse([
			{
				id: "vendor/new-thinker",
				modality: "chat",
				context_length: 400_000,
				max_completion_tokens: 32_000,
				input_modalities: ["text", "image"],
				supported_parameters: ["tools", "tool_choice", "reasoning"],
				pricing: { prompt: "1.00", completion: "4.00", unit: "1M tokens" },
			},
			{
				id: "vendor/new-chat",
				modality: "chat",
				context_length: 128_000,
				max_completion_tokens: 8_192,
				input_modalities: ["text"],
				supported_parameters: [],
			},
			// Capabilities not recorded yet: no claims either way.
			{ id: "vendor/unrecorded", modality: "chat" },
		]);

		const models = await routerbaseModelManagerOptions({
			apiKey: "sk-rb-key",
			fetch: fetchMock,
		}).fetchDynamicModels?.();
		const byId = new Map(models?.map(model => [model.id, model]));

		const thinker = byId.get("vendor/new-thinker");
		expect(thinker?.contextWindow).toBe(400_000);
		expect(thinker?.maxTokens).toBe(32_000);
		expect(thinker?.input).toEqual(["text", "image"]);
		expect(thinker?.reasoning).toBe(true);
		expect(thinker?.supportsTools).toBe(true);
		// List price is not the charge; an unknown tariff stays unpriced.
		expect(thinker?.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });

		const chat = byId.get("vendor/new-chat");
		expect(chat?.reasoning).toBe(false);
		expect(chat?.supportsTools).toBe(false);
		expect(chat?.input).toEqual(["text"]);

		const unrecorded = byId.get("vendor/unrecorded");
		expect(unrecorded?.reasoning).toBe(false);
		expect(unrecorded?.supportsTools).toBeUndefined();
		expect(unrecorded?.contextWindow).toBeNull();
	});

	test("discovery merges additively: seed rows the live roster omits stay selectable", async () => {
		const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-catalog-routerbase-merge-"));
		const options = {
			...routerbaseModelManagerOptions({
				apiKey: "sk-rb-key",
				fetch: modelsResponse([
					{ id: DEFAULT_MODEL, modality: "chat" },
					{ id: "vendor/new-chat", modality: "chat", supported_parameters: ["tools"] },
				]),
			}),
			staticModels: getBundledModels("routerbase"),
			cacheDbPath: path.join(tempDir, "models.db"),
		};

		try {
			expect(options.dynamicModelsAuthoritative).toBeUndefined();

			const result = await resolveProviderModels(options, "online");
			const ids = result.models.map(model => model.id);

			expect(ids).toContain("vendor/new-chat");
			expect(ids).toContain(DEFAULT_MODEL);
			expect(ids).toContain("zai/glm-5.2");
			expect(result.stale).toBe(false);
		} finally {
			await fs.rm(tempDir, { recursive: true, force: true });
		}
	});

	test("modality classification keeps chat and unknown rows, drops media and embeddings", () => {
		expect(isRouterbaseChatModality("chat")).toBe(true);
		expect(isRouterbaseChatModality(undefined)).toBe(true);
		expect(isRouterbaseChatModality(42)).toBe(true);
		for (const modality of ["image", "video", "audio", "embedding"]) {
			expect(isRouterbaseChatModality(modality)).toBe(false);
		}
	});
});
