import { LibraryProvider } from "@/components/library-provider";
import { Shell } from "@/components/shell";
export default function VaultLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <LibraryProvider>
      <Shell>{children}</Shell>
    </LibraryProvider>
  );
}
