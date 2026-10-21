import { requireCollector } from "@/server/auth";
import { handleApi } from "@/server/http";
import { photoResponse } from "@/server/photos";

export const runtime = "nodejs";
export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    return photoResponse(collector.id, (await context.params).id);
  });
}
