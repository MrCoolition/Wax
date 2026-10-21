import { requireCollector } from "@/server/auth";
import { shareCrate, unshareCrate } from "@/server/collection";
import { handleApi } from "@/server/http";
type Context = { params: Promise<{ id: string }> };
export async function POST(request: Request, context: Context) {
  return handleApi(request, async () => {
    const user = await requireCollector();
    const { email } = await request.json();
    return shareCrate(user.id, (await context.params).id, email);
  });
}
export async function DELETE(request: Request, context: Context) {
  return handleApi(request, async () => {
    const user = await requireCollector();
    const { userId } = await request.json();
    return unshareCrate(user.id, (await context.params).id, userId);
  });
}
