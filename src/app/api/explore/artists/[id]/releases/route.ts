import { requireCollector } from "@/server/auth";
import {
  explorerIdSchema,
  explorerPageSchema,
  fetchExplorerDiscography,
} from "@/server/explorer";
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
    const { page } = explorerPageSchema.parse(
      Object.fromEntries(new URL(request.url).searchParams),
    );
    await consumeIntegrationLimit(collector.id, "explorer", 120, 60);
    return fetchExplorerDiscography(artistId, page);
  });
  response.headers.set("Cache-Control", "private, no-store");
  return response;
}
