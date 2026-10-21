import { RecordDetail } from "@/components/record-detail";
export default async function RecordPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <RecordDetail id={id} />;
}
