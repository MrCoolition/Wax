import { requireCollector } from "@/server/auth";
import { getCopy, updateCopy, deleteCopy } from "@/server/collection";
import { handleApi } from "@/server/http";
type Context = { params: Promise<{ id: string }> };
export async function GET(request: Request, context: Context) {
  return handleApi(request, async () =>
    getCopy((await requireCollector()).id, (await context.params).id),
  );
}
export async function PATCH(request: Request, context: Context) {
  return handleApi(request, async () => {
    const user = await requireCollector();
    return updateCopy(user.id, (await context.params).id, await request.json());
  });
}
export async function DELETE(request: Request, context: Context) {
  return handleApi(request, async () => {
    const user = await requireCollector();
    const query = new URL(request.url).searchParams.get("version");
    const version = query ? Number(query) : (await request.json()).version;
    return deleteCopy(user.id, (await context.params).id, version);
  });
}
