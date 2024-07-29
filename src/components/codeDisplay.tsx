'use client';

import { IconCheck, IconCopy, IconDownload } from '@tabler/icons-react';
import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface CodeDisplayProps {
	code: string;
	language: string;
	fileName: string;
}

export default function CodeDisplay({ code, language, fileName }: CodeDisplayProps) {
	const [copySuccess, setCopySuccess] = useState(false);

	function handleCopy() {
		navigator.clipboard.writeText(code).then(() => {
			setCopySuccess(true);
			setTimeout(() => setCopySuccess(false), 2000);
		});
	}

	function handleDownload() {
		const element = document.createElement('a');
		const file = new Blob([code], { type: 'text/plain' });
		element.href = URL.createObjectURL(file);
		element.download = fileName;
		document.body.appendChild(element);
		element.click();
		document.body.removeChild(element);
	}

	return (
		<div className="border border-gray-700 rounded-md overflow-hidden h-fit">
			<SyntaxHighlighter
				language={language}
				style={atomDark}
				codeTagProps={{ className: 'text-sm' }}
				customStyle={{ margin: 0, borderRadius: 0 }}>
				{code}
			</SyntaxHighlighter>
			<div className="bg-[#2a2c30] flex justify-end items-center p-2 gap-2">
				<button
					onClick={handleCopy}
					className="text-gray-300 hover:text-white p-1 rounded transition-colors duration-200"
					title={copySuccess ? 'Copied!' : 'Copy code'}>
					{copySuccess ? <IconCheck size={20} /> : <IconCopy size={20} />}
				</button>
				<button
					onClick={handleDownload}
					className="text-gray-300 hover:text-white p-1 rounded transition-colors duration-200"
					title="Download code">
					<IconDownload size={20} />
				</button>
			</div>
		</div>
	);
}
