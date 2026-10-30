"""Render the README's illustrative GPU tool loop (CPU-only, Pillow 12.0.0).

Run from the repository root:
    uv run --no-sync --with pillow==12.0.0 python tools/render_hero.py

The animation illustrates control/data flow; its timing is not a measurement.
"""

from functools import lru_cache
from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, SCALE = 1440, 500, 2
FPS, SECONDS = 12, 10
BG = (5, 9, 12)
WHITE = (237, 243, 240)
GREEN = (174, 247, 79)
CYAN = (94, 219, 225)
MUTED = (137, 157, 158)
DIM = (66, 87, 89)
ROBOTO = Path('/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF')
MONO = Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')


@lru_cache
def font(size, kind='regular'):
    paths = {
        'regular': ROBOTO / 'Roboto-Regular.ttf',
        'light': ROBOTO / 'Roboto-Light.ttf',
        'medium': ROBOTO / 'Roboto-Medium.ttf',
        'mono': MONO,
    }
    path = paths[kind]
    if not path.is_file():
        raise FileNotFoundError(f'Install Roboto and DejaVu fonts: {path}')
    return ImageFont.truetype(str(path), round(size * SCALE))


def box(draw, bounds, fill=None, outline=None, radius=0, width=1):
    bounds = tuple(round(v * SCALE) for v in bounds)
    draw.rounded_rectangle(bounds, radius=radius * SCALE, fill=fill,
                           outline=outline, width=width * SCALE)


def line(draw, points, color, width=1):
    draw.line([(round(x*SCALE), round(y*SCALE)) for x,y in points],
              fill=color, width=width*SCALE)


def text(draw, xy, value, size=16, color=WHITE, kind='regular'):
    draw.text(tuple(round(v*SCALE) for v in xy), value,
              font=font(size, kind), fill=color, anchor='lt')
    return draw.textlength(value, font=font(size, kind)) / SCALE


def mix(a, b, t):
    return tuple(round(x*(1-t)+y*t) for x,y in zip(a,b))


def particle(draw, x, y, color):
    for radius, strength in ((12,.07),(8,.13),(5,.25),(2.5,1)):
        draw.ellipse(((x-radius)*SCALE,(y-radius)*SCALE,
                      (x+radius)*SCALE,(y+radius)*SCALE),
                     fill=mix(BG,color,strength))


def base_image():
    im = Image.new('RGB', (WIDTH*SCALE, HEIGHT*SCALE), BG)
    glow = Image.new('RGB', im.size, BG)
    gd = ImageDraw.Draw(glow)
    gd.ellipse((780*SCALE,40*SCALE,1410*SCALE,440*SCALE), fill=(9,28,24))
    im = glow.filter(ImageFilter.GaussianBlur(95*SCALE))
    d = ImageDraw.Draw(im)
    for x in range(0, WIDTH, 32): line(d, [(x,0),(x,HEIGHT)], (15,23,26))
    for y in range(0, HEIGHT, 32): line(d, [(0,y),(WIDTH,y)], (15,23,26))
    text(d,(58,49),'RISC-V  ×  CUDA',14,MUTED,'mono')
    w = text(d,(52,105),'RISC',136,WHITE,'light')
    text(d,(52+w,105),'uda',136,GREEN,'light')
    text(d,(61,266),'A RISC-V core.',31,WHITE,'light')
    text(d,(61,307),'Inside your GPU.',31,WHITE,'light')
    box(d,(60,368,309,398),(15,26,20),(45,66,40),14)
    particle(d,76,383,GREEN)
    text(d,(92,377),'ONE HOST GRAPH LAUNCH',12,GREEN,'mono')
    text(d,(61,416),'Qwen → Python → RISC-V → Qwen',14,MUTED,'mono')

    # One GPU boundary contains both the model and guest interpreter.
    box(d,(665,54,1380,405),(8,17,20),(47,71,66),18)
    line(d,[(685,54),(774,54)],GREEN,2)
    text(d,(691,75),'GPU-RESIDENT TOOL LOOP',15,GREEN,'mono')
    text(d,(1271,76),'RV32IM',14,MUTED,'mono')
    for x in range(698,1360,22):
        line(d,[(x,46),(x,53)],(48,69,63))
        line(d,[(x,406),(x,413)],(48,69,63))
    for y in range(87,384,22):
        line(d,[(657,y),(664,y)],(48,69,63))
        line(d,[(1381,y),(1388,y)],(48,69,63))

    box(d,(693,112,959,286),(11,23,29),(45,87,94),12)
    text(d,(715,133),'QWEN',23,WHITE,'medium')
    text(d,(866,139),'LLM',12,CYAN,'mono')
    for row in range(5):
        y=179+row*17
        box(d,(715,y,937,y+9),(15,35,42),(33,66,74),3)

    box(d,(1041,112,1352,286),(15,25,20),(72,100,48),12)
    text(d,(1063,133),'RISC-V',23,WHITE,'medium')
    text(d,(1212,139),'ON CUDA',12,GREEN,'mono')
    text(d,(1063,173),'PYTHON INTERPRETER',12,MUTED,'mono')
    box(d,(1062,198,1330,238),(7,16,13),(37,60,39),6)
    text(d,(1064,258),'FETCH  /  DECODE  /  EXECUTE',11,DIM,'mono')

    line(d,[(961,190),(1037,190)],(47,83,76),2)
    line(d,[(1030,184),(1037,190),(1030,196)],(77,120,101),2)
    line(d,[(1040,252),(963,252)],(47,83,76),2)
    line(d,[(970,246),(963,252),(970,258)],(77,120,101),2)
    text(d,(977,169),'code',11,MUTED,'mono')
    text(d,(973,270),'result',11,MUTED,'mono')

    box(d,(693,307,1352,381),(6,13,17),(29,48,52),9)
    text(d,(714,321),'USER',11,MUTED,'mono')
    text(d,(764,317),'What is 2 × 8?',20,WHITE)
    line(d,[(714,352),(1331,352)],(26,44,46))
    line(d,[(0,454),(WIDTH,454)],(35,52,48))
    return im


def frame(base, time):
    im=base.copy(); d=ImageDraw.Draw(im)
    phase=min(int(time/2.5),3)
    local=(time%2.5)/2.5
    # Decorative model activations; no trace or performance numbers are implied.
    for row in range(5):
        x=721+((time*.21+row*.17)%1)*202
        y=183+row*17
        color=CYAN if phase in (0,3) else mix(BG,CYAN,.2)
        box(d,(x,y-2,x+7,y+2),color,radius=2)
    source='print(2 * 8)'
    visible=source[:max(0,min(len(source),int(local*20)))] if phase==0 else source
    text(d,(1076,210),visible,20,GREEN,'mono')
    if phase==0 and local<.8:
        x=1076+len(visible)*font(20,'mono').getlength('0')/SCALE
        box(d,(x+2,209,x+4,229),GREEN)
    if phase==1:
        # Small instruction activity ticks stay within the interpreter card.
        for i in range(16):
            color=GREEN if (i+int(time*12))%5<2 else (37,57,33)
            box(d,(1064+i*16,246,1074+i*16,249),color,radius=1)
    if phase in (0,1): particle(d,968+((time*1.2)%1)*65,190,CYAN)
    if phase==2:
        particle(d,1032-local*64,252,GREEN)
        box(d,(986,240,1017,263),(15,30,19),(80,111,52),5)
        text(d,(991,244),'16',15,GREEN,'mono')
    labels=['01  GENERATING PYTHON','02  EXECUTING ON RISC-V',
            '03  RETURNING TOOL RESULT','04  ANSWERING THE USER']
    color=CYAN if phase in (0,3) else GREEN
    particle(d,720,367,color)
    text(d,(736,361),labels[phase],11,color,'mono')
    for i in range(4):
        box(d,(1236+i*24,363,1252+i*24,367),color if i<=phase else (34,49,48),radius=2)
    if phase==3:
        box(d,(1007,313,1337,345),(15,28,20),(64,91,45),6)
        text(d,(1021,322),'Qwen: The answer is 16.',15,GREEN,'mono')
    tokens=['USER','2 × 8?','QWEN','print','(','2','*','8',')','RV32IM','stdout','16','QWEN','The answer is 16.']
    x=22
    active=(int(time/SECONDS*len(tokens)))%len(tokens)
    for i,token in enumerate(tokens):
        w=ImageDraw.Draw(im).textlength(token,font=font(13,'mono'))/SCALE+24
        accent=GREEN if i in (9,10,11) else CYAN
        box(d,(x,466,x+w,491),(12,23,23),accent if i==active else (34,53,53),5)
        text(d,(x+12,473),token,13,accent if i==active else MUTED,'mono')
        x+=w+8
    return im.resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/media')
    parser.add_argument('--preview',type=Path,help='Optional contact sheet outside the repository')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    base=base_image()
    samples=[frame(base,t) for t in (1.5,3.5,6.5,8.5)]
    # A shared palette prevents color flicker between frames.
    sheet=Image.new('RGB',(WIDTH,HEIGHT*4))
    for i,im in enumerate(samples): sheet.paste(im,(0,i*HEIGHT))
    palette=sheet.quantize(colors=192,method=Image.Quantize.MEDIANCUT)
    # Reserve accent shades: small moving cyan/green dots otherwise lose their
    # hue when the palette is dominated by the dark background and white type.
    colors=palette.getpalette()
    accents=[mix(BG,color,strength) for color in (CYAN,GREEN)
             for strength in (0,.07,.13,.2,.25,.4,.6,.8,1)]
    colors[192*3:192*3+len(accents)*3]=[c for rgb in accents for c in rgb]
    palette.putpalette(colors)
    frames=[frame(base,i/FPS).quantize(palette=palette,dither=Image.Dither.NONE)
            for i in range(FPS*SECONDS)]
    durations=[round((i+1)*100/FPS)*10-round(i*100/FPS)*10 for i in range(len(frames))]
    frames[0].save(args.output/'riscuda-hero.gif',save_all=True,append_images=frames[1:],
                   duration=durations,loop=0,optimize=True,disposal=1)
    samples[-1].save(args.output/'riscuda-hero.png',optimize=True)
    if args.preview:
        sheet.resize((1008,1400),Image.Resampling.LANCZOS).save(args.preview)
    print(f'GIF: {len(frames)} frames, {sum(durations)/1000:g}s, '
          f'{(args.output/"riscuda-hero.gif").stat().st_size/1048576:.2f} MiB')


if __name__=='__main__':
    main()
