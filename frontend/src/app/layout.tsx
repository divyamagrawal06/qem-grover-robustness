import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Grover Robustness Dashboard",
  description:
    "Interactive analysis of Grover algorithm failure regimes with deep-learning quantum error mitigation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
