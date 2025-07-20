import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AI for Landlords',
  description: 'Hi! I am your AI Assistant for Landlords. Ask me about rent estimation, tenant screening, or predictive maintenance! (Notifications and alerts UI have been removed.)',
  generator: 'v0.dev',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
