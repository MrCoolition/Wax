import type { Metadata } from "next";
import { ArtistExplorer } from "@/components/artist-explorer";

export const metadata: Metadata = { title: "Artist Tree · WAX" };

export default async function RecordExplorerPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <ArtistExplorer key={id} copyId={id} />;
}
