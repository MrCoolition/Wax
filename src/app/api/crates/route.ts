import { requireCollector } from "@/server/auth";
import { createCrate } from "@/server/collection";
import { handleApi } from "@/server/http";
export async function POST(request: Request) {
  return handleApi(request, async () =>
    createCrate((await requireCollector()).id, await request.json()),
  );
}
