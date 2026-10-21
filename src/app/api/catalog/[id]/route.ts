import { z } from "zod";
import { requireCollector } from "@/server/auth";
import { handleApi } from "@/server/http";
import { fetchRelease, releaseQuerySchema } from "@/server/catalog";
import { consumeIntegrationLimit } from "@/server/integration-common";

export const runtime = "nodejs";
export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const { id } = await context.params;
    const releaseId = z.coerce
      .number()
      .int()
      .positive()
      .max(Number.MAX_SAFE_INTEGER)
      .parse(id);
    const options = releaseQuerySchema.parse(
      Object.fromEntries(new URL(request.url).searchParams),
    );
    await consumeIntegrationLimit(collector.id, "catalog", 120, 60);
    return fetchRelease(releaseId, { refresh: options.refresh === "true" });
  });
}
