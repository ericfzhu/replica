import { GeistMono } from 'geist/font/mono';
import type { Metadata } from 'next';

import { cn } from '@/lib/utils';

import './globals.css';

export const metadata: Metadata = {
	title: 'REPLICA: Implementations of machine learning papers in PyTorch',
	description: 'Implementations of machine learning papers in PyTorch',
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang="en">
			<link rel="icon" href="/favicon.ico" />
			<body className={cn(GeistMono.className)}>{children}</body>
		</html>
	);
}
