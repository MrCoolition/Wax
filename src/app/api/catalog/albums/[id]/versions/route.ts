import { requireCollector } from "@/server/auth";
import {
  albumVersionsSchema,
  catalogIdSchema,
  fetchAlbumVersions,
} from "@/server/catalog";
import { handleApi } from "@/server/http";
import { consumeIntegrationLimit } from "@/server/integration-common";

export const runtime = "nodejs";
export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    const masterId = catalogIdSchema.parse((await context.params).id);
    const search = albumVersionsSchema.parse(
      Object.fromEntries(new URL(request.url).searchParams),
    );
    await consumeIntegrationLimit(collector.id, "catalog", 120, 60);
    return fetchAlbumVersions(masterId, search);
  });
}
