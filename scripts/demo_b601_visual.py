"""Render a visual-model demo; it is not a benchmark trial."""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from tinyvla.language import parse
from tinyvla.sim_b601_visual import B601VisualSim


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'portfolio/media/b601-rs-visual-proxy-demo.mp4'


def main():
    sim = B601VisualSim(size=384)
    task = parse('put the red cylinder on the yellow pad')
    try:
        sim.reset(seed=360000, suite='nominal')
        frames = [sim.image(384, True) for _ in range(12)]
        sim.execute_chunk(sim.teacher(task), frames)
        outcome = sim.score(task)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(OUTPUT, fps=20) as writer:
            for rgb in frames:
                canvas = Image.new('RGB', (384, 432), (20, 25, 35))
                canvas.paste(Image.fromarray(rgb), (0, 48))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 7), 'B601-RS visual proxy | Panda physics | scripted teacher', fill='white')
                draw.text((8, 26), task.instruction, fill=(175, 225, 255))
                writer.append_data(np.asarray(canvas))
        print({'video': str(OUTPUT), **outcome})
    finally:
        sim.close()


if __name__ == '__main__':
    main()
