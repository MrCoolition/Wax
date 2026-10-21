import { requireCollector } from "@/server/auth";
import { createCopy } from "@/server/collection";
import { handleApi } from "@/server/http";
export async function POST(request: Request) {
  return handleApi(request, async () => {
    const user = await requireCollector();
    return createCopy(user.id, await request.json());
  });
}
