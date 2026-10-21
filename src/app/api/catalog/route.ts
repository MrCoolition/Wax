import { requireCollector } from "@/server/auth";
import { handleApi } from "@/server/http";
import { catalogSearchSchema, searchCatalog } from "@/server/catalog";
import { consumeIntegrationLimit } from "@/server/integration-common";

export const runtime = "nodejs";
export async function GET(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const search = catalogSearchSchema.parse(
      Object.fromEntries(new URL(request.url).searchParams),
    );
    await consumeIntegrationLimit(collector.id, "catalog", 120, 60);
    return { candidates: await searchCatalog(search) };
  });
}
