// 注册渲染用字体（全部 SIL OFL 1.1，许可文件在 fonts/OFL-*.txt）。成片渲染、单帧预览、剖析脚本共用。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'fonts');
const FONTS = [
  ['Anton-Regular.ttf', 'Anton'],
  ['NotoSansSC-Medium.otf', 'Noto Sans SC'],
  ['NotoSansSC-Bold.otf', 'Noto Sans SC'],
  ['NotoSansSC-Black.otf', 'Noto Sans SC'],
  ['Inter-ExtraBold.ttf', 'Inter'], // 由 Inter 可变字体切出的 ExtraBold（wght 800, opsz 32）
  ['Barlow-Bold.ttf', 'Barlow'],
  ['BebasNeue-Regular.ttf', 'Bebas Neue'],
  ['Poppins-Bold.ttf', 'Poppins'],
  ['SmileySans-Oblique.ttf', 'Smiley Sans'], // 得意黑
  ['ZCOOLQingKeHuangYou-Regular.ttf', 'ZCOOL QingKe HuangYou'], // 站酷庆科黄油体
  ['BarlowCondensed-BoldItalic.ttf', 'Barlow Condensed'],
];

export function registerFonts(GlobalFonts) {
  for (const [file, family] of FONTS) {
    const p = path.join(DIR, file);
    if (fs.existsSync(p)) GlobalFonts.registerFromPath(p, family);
  }
}
