import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import axios from "axios";

// Mock axios before importing the module under test
vi.mock("axios", async () => {
  const actual = await vi.importActual<typeof import("axios")>("axios");
  return {
    ...actual,
    default: {
      ...actual.default,
      create: vi.fn(() => ({
        get: vi.fn(),
        post: vi.fn(),
        delete: vi.fn(),
      })),
    },
  };
});

describe("API client construction", () => {
  it("creates axios instance with correct baseURL default", () => {
    // Re-import to trigger module-level axios.create()
    vi.resetModules();
    delete process.env.NEXT_PUBLIC_API_URL;

    const createSpy = vi.spyOn(axios, "create");
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require("@/lib/api");

    expect(createSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        baseURL: "http://localhost:8000/api/v1",
        timeout: 30_000,
      })
    );
  });

  it("uses NEXT_PUBLIC_API_URL env var when set", () => {
    vi.resetModules();
    process.env.NEXT_PUBLIC_API_URL = "http://custom-api:9000/api/v1";
    const createSpy = vi.spyOn(axios, "create");

    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require("@/lib/api");

    expect(createSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        baseURL: "http://custom-api:9000/api/v1",
      })
    );

    delete process.env.NEXT_PUBLIC_API_URL;
  });
});

describe("corpusApi", () => {
  let mockClient: { get: ReturnType<typeof vi.fn>; post: ReturnType<typeof vi.fn>; delete: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    vi.resetModules();
    mockClient = { get: vi.fn(), post: vi.fn(), delete: vi.fn() };
    vi.spyOn(axios, "create").mockReturnValue(mockClient as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("list() calls GET /corpus/", async () => {
    mockClient.get.mockResolvedValue({ data: [] });
    const { corpusApi } = await import("@/lib/api");
    await corpusApi.list();
    expect(mockClient.get).toHaveBeenCalledWith("/corpus/");
  });

  it("get(id) calls GET /corpus/:id", async () => {
    mockClient.get.mockResolvedValue({ data: { id: "abc" } });
    const { corpusApi } = await import("@/lib/api");
    await corpusApi.get("abc");
    expect(mockClient.get).toHaveBeenCalledWith("/corpus/abc");
  });

  it("delete(id) calls DELETE /corpus/:id", async () => {
    mockClient.delete.mockResolvedValue({ data: undefined });
    const { corpusApi } = await import("@/lib/api");
    await corpusApi.delete("abc");
    expect(mockClient.delete).toHaveBeenCalledWith("/corpus/abc");
  });

  it("upload() calls POST /corpus/upload with multipart header", async () => {
    mockClient.post.mockResolvedValue({ data: { id: "new" } });
    const { corpusApi } = await import("@/lib/api");
    const form = new FormData();
    await corpusApi.upload(form);
    expect(mockClient.post).toHaveBeenCalledWith(
      "/corpus/upload",
      form,
      expect.objectContaining({ headers: { "Content-Type": "multipart/form-data" } })
    );
  });
});

describe("experimentsApi", () => {
  let mockClient: { get: ReturnType<typeof vi.fn>; post: ReturnType<typeof vi.fn>; delete: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    vi.resetModules();
    mockClient = { get: vi.fn(), post: vi.fn(), delete: vi.fn() };
    vi.spyOn(axios, "create").mockReturnValue(mockClient as any);
  });

  it("list() calls GET /experiments/", async () => {
    mockClient.get.mockResolvedValue({ data: [] });
    const { experimentsApi } = await import("@/lib/api");
    await experimentsApi.list();
    expect(mockClient.get).toHaveBeenCalledWith("/experiments/");
  });

  it("getRuns(id) calls GET /experiments/:id/runs", async () => {
    mockClient.get.mockResolvedValue({ data: [] });
    const { experimentsApi } = await import("@/lib/api");
    await experimentsApi.getRuns("exp-1");
    expect(mockClient.get).toHaveBeenCalledWith("/experiments/exp-1/runs");
  });

  it("getRunTrace() calls correct endpoint", async () => {
    mockClient.get.mockResolvedValue({ data: {} });
    const { experimentsApi } = await import("@/lib/api");
    await experimentsApi.getRunTrace("exp-1", "run-2");
    expect(mockClient.get).toHaveBeenCalledWith("/experiments/exp-1/runs/run-2/trace");
  });
});

describe("evaluationApi", () => {
  let mockClient: { get: ReturnType<typeof vi.fn>; post: ReturnType<typeof vi.fn>; delete: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    vi.resetModules();
    mockClient = { get: vi.fn(), post: vi.fn(), delete: vi.fn() };
    vi.spyOn(axios, "create").mockReturnValue(mockClient as any);
  });

  it("getLeaderboard() without args hits /evaluation/leaderboard", async () => {
    mockClient.get.mockResolvedValue({ data: [] });
    const { evaluationApi } = await import("@/lib/api");
    await evaluationApi.getLeaderboard();
    expect(mockClient.get).toHaveBeenCalledWith(
      "/evaluation/leaderboard",
      expect.objectContaining({ params: { experiment_id: undefined } })
    );
  });

  it("getMetrics(runId) hits correct endpoint", async () => {
    mockClient.get.mockResolvedValue({ data: {} });
    const { evaluationApi } = await import("@/lib/api");
    await evaluationApi.getMetrics("run-99");
    expect(mockClient.get).toHaveBeenCalledWith("/evaluation/run/run-99/metrics");
  });
});
