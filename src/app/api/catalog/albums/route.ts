import { requireCollector } from "@/server/auth";
import { albumSearchSchema, searchAlbums } from "@/server/catalog";
import { handleApi } from "@/server/http";
import { consumeIntegrationLimit } from "@/server/integration-common";

export const runtime = "nodejs";
export async function GET(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const search = albumSearchSchema.parse(
      Object.fromEntries(new URL(request.url).searchParams),
    );
    await consumeIntegrationLimit(collector.id, "catalog", 120, 60);
    return searchAlbums(search);
  });
}
