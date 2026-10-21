import { requireCollector } from "@/server/auth";
import { logListen } from "@/server/collection";
import { handleApi } from "@/server/http";
export async function POST(request: Request) {
  return handleApi(request, async () =>
    logListen((await requireCollector()).id, await request.json()),
  );
}
