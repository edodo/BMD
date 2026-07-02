python -c "
import numpy as np, glob, os
from PIL import Image
md='dataset-dcm/test/gather_mask/SegmentationClass'
f=sorted(glob.glob(os.path.join(r'c:\Users\csm02\Desktop\edward\bmd\src\BMD\model',md,'*.png')))[0]
im=Image.open(f); print('file:',os.path.basename(f)); print('mode:',im.mode,'size:',im.size)
a=np.array(im); print('array shape:',a.shape,'dtype:',a.dtype)
print('unique values:',np.unique(a)[:30])
if im.mode=='P':
    pal=np.array(im.getpalette()).reshape(-1,3)[:8]; print('palette[:8]:',pal.tolist())
print('value counts:',[(int(v),int((a==v).sum())) for v in np.unique(a)][:12] if a.ndim==2 else 'RGB')
"