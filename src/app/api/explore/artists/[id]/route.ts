import { requireCollector } from "@/server/auth";
import { explorerIdSchema, fetchExplorerArtist } from "@/server/explorer";
import { handleApi } from "@/server/http";
import { consumeIntegrationLimit } from "@/server/integration-common";

export const runtime = "nodejs";
export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const response = await handleApi(request, async () => {
    const collector = await requireCollector();
    const { id } = await context.params;
    const artistId = explorerIdSchema.parse(id);
    await consumeIntegrationLimit(collector.id, "explorer", 120, 60);
    return fetchExplorerArtist(artistId);
  });
  response.headers.set("Cache-Control", "private, no-store");
  return response;
}
