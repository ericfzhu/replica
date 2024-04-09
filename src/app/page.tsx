import Image from 'next/image';
import Link from 'next/link';

export default function Home() {
	return (
		<main className="flex min-h-screen flex-col items-center gap-24 bg-white p-24">
			<section className="flex flex-col items-center gap-12">
				<h1 className="flex-wrap text-center text-4xl text-black">
					Imitator: <span className="text-accent-500">Im</span>age-to-<span className="text-accent-500">I</span>mage{' '}
					<span className="text-accent-500">T</span>ransl<span className="text-accent-500">at</span>i
					<span className="text-accent-500">o</span>n with a <span className="text-accent-500">R</span>esNeXt-based GAN
				</h1>

				<div className="flex gap-2">
					<Link
						href="https://github.com/ericfzhu/imitator"
						className="bg-accent-500 border-accent-500 hover:bg-accent-400 rounded-full border-[1px] px-4 py-2 text-white duration-300">
						Code
					</Link>
					<Link
						href="https://huggingface.co/ericfzhu/imitator"
						className="bg-accent-500 border-accent-500 hover:bg-accent-400 rounded-full border-[1px] px-4 py-2 text-white duration-300">
						HuggingFace
					</Link>
				</div>
			</section>

			<section>
				<h2 className="text-3xl">Demo</h2>
			</section>

			<section>
				<h2 className="text-3xl">Abstract</h2>
			</section>

			<section>
				<h2 className="text-3xl">Architecture</h2>
			</section>
		</main>
	);
}
