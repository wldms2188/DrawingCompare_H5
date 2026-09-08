from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import re
import cv2
import numpy as np

@dataclass(frozen=True)
class Box:
    x:int; y:int; w:int; h:int
    def xyxy(self): return self.x,self.y,self.x+self.w,self.y+self.h
    def pad(self,p:int,W:int,H:int):
        x=max(0,self.x-p); y=max(0,self.y-p); xx=min(W,self.x+self.w+p); yy=min(H,self.y+self.h+p)
        return Box(x,y,max(1,xx-x),max(1,yy-y))

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0
    region_type:str='general_change'; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None
    difference_crop:Optional[np.ndarray]=None; old_text:str=''; new_text:str=''; change_kind:str='changed_value'

@dataclass
class ChangeDetectionResult:
    success:bool
    regions:List[ChangeRegion]=field(default_factory=list)
    difference_image:Optional[np.ndarray]=None
    threshold_image:Optional[np.ndarray]=None
    change_pixel_ratio:float=0.0
    reason:str=''
    @property
    def region(self): return self.regions

class ChangeDetector:
    """Vector/native semantic comparison plus structural line correspondence."""
    def __init__(self, config=None):
        self.max_word_distance_ratio=.035
        self.merge_gap=24
        self.min_line_length_ratio=.018
        self.line_position_ratio=.035
        self.line_angle_deg=9.0
        self.line_length_ratio=.30

    @staticmethod
    def _img(page):
        if isinstance(page,np.ndarray): return np.asarray(page)
        if hasattr(page,'image'): return np.asarray(page.image)
        raise TypeError('페이지 이미지 배열을 찾을 수 없습니다.')

    @staticmethod
    def _gray(img):
        if img.ndim==2: return img.astype(np.uint8)
        if img.shape[2]==4: return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _crop(img,b):
        H,W=img.shape[:2]; x=max(0,min(W-1,b.x)); y=max(0,min(H-1,b.y)); xx=max(x+1,min(W,b.x+b.w)); yy=max(y+1,min(H,b.y+b.h)); return img[y:yy,x:xx]

    @staticmethod
    def _norm(t): return re.sub(r'\s+','',str(t).upper().replace('−','-').replace('–','-').replace('—','-'))

    @staticmethod
    def _class(text,context=''):
        u=str(text).strip().upper(); c=str(context).upper()
        if re.search(r'NOTE|NOTES|MATERIAL|FINISH|DEBURR|BURR|UNLESS|REMARK|COMMENT',c): return 'NOTE'
        if re.search(r'POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|POS',c): return 'GDT'
        if re.search(r'±|Ø|⌀|\bR\s*\d|^\d+(?:\.\d+)?$',u): return 'DIMENSION'
        return 'TEXT'

    @staticmethod
    def _value_like(text):
        u=str(text).strip().upper()
        if not u: return False
        if re.search(r'±|Ø|⌀|^R\s*[-+]?\d|^[-+]?\d+(?:\.\d+)?$',u): return True
        if re.fullmatch(r'[A-Z]{1,8}[-_/]?[A-Z0-9]{2,12}',u) and re.search(r'\d',u): return True
        return False

    def _words(self,page):
        try:
            import fitz
            doc=fitz.open(page.pdf_path); p=doc.load_page(int(page.page_index)); pw,ph=p.rect.width,p.rect.height
            raw=p.get_text('words'); line_map={}
            for z in raw:
                if len(z)>=8: line_map.setdefault((z[5],z[6]),[]).append(str(z[4]).strip())
            out=[]
            for z in raw:
                if len(z)<8 or not str(z[4]).strip(): continue
                x0,y0,x1,y1,text,block,line,word=z[:8]; context=' '.join(line_map.get((block,line),[text]))
                out.append({'text':str(text),'context':context,'block':int(block),'line':int(line),'x':x0/pw,'y':y0/ph,'w':(x1-x0)/pw,'h':(y1-y0)/ph,'cx':((x0+x1)/2)/pw,'cy':((y0+y1)/2)/ph,'class':self._class(text,context)})
            doc.close(); return out
        except Exception: return []

    @staticmethod
    def _affine_point(x,y,M):
        p=np.array([x,y,1.],dtype=np.float32); q=np.asarray(M,dtype=np.float32).reshape(2,3)@p; return float(q[0]),float(q[1])

    def _mapped_words(self,words,after_shape,view_shape,M):
        H,W=after_shape[:2]; out=[]
        for q in words:
            pts=[self._affine_point(q['x']*W,q['y']*H,M),self._affine_point((q['x']+q['w'])*W,(q['y']+q['h'])*H,M)]
            x=min(a[0] for a in pts); y=min(a[1] for a in pts); xx=max(a[0] for a in pts); yy=max(a[1] for a in pts)
            out.append({**q,'x':x,'y':y,'w':max(1,xx-x),'h':max(1,yy-y),'cx':(x+xx)/2,'cy':(y+yy)/2})
        return out

    @staticmethod
    def _box(q,pad,W,H):
        x=int(round(q['x'])); y=int(round(q['y'])); xx=int(round(q['x']+q['w'])); yy=int(round(q['y']+q['h']))
        return Box(x,y,max(1,xx-x),max(1,yy-y)).pad(pad,W,H)

    @staticmethod
    def _iou(a,b):
        ax,ay,axx,ayy=a.xyxy(); bx,by,bxx,byy=b.xyxy(); iw=max(0,min(axx,bxx)-max(ax,bx)); ih=max(0,min(ayy,byy)-max(ay,by)); inter=iw*ih; return inter/max(1,a.w*a.h+b.w*b.h-inter)

    def _pair_words(self,old,new,W,H):
        maxd=max(20,min(W,H)*self.max_word_distance_ratio); cand=[]
        for i,o in enumerate(old):
            for j,n in enumerate(new):
                if o['class']!=n['class']: continue
                d=float(np.hypot(o['cx']-n['cx'],o['cy']-n['cy']))
                if d>maxd: continue
                size=max(o['h'],n['h'])/max(1,min(o['h'],n['h']))
                if size>2.2: continue
                text_bonus=0 if self._norm(o['text'])==self._norm(n['text']) else -0.25
                cand.append((d/maxd+text_bonus,i,j))
        cand.sort(); used_o=set(); used_n=set(); pairs=[]
        for score,i,j in cand:
            if i in used_o or j in used_n: continue
            used_o.add(i); used_n.add(j); pairs.append((old[i],new[j],score))
        return pairs,used_o,used_n

    def _add_region(self,regions,before,after,b,kind,old_text='',new_text='',confidence=.65):
        blank=np.full((b.h,b.w,3),255,np.uint8)
        regions.append(ChangeRegion(b.x,b.y,b.w,b.h,b.w*b.h,0.0,kind,confidence,self._crop(before,b) if old_text else blank.copy(),self._crop(after,b) if new_text else blank.copy(),None,old_text,new_text,kind))

    def _expand_note_box(self,seed,words,W,H,pad=14):
        """Return one semantic NOTE block around a changed NOTE token."""
        sx,sy=seed['cx'],seed['cy']; selected=[]
        for q in words:
            if q['class']!='NOTE': continue
            d=float(np.hypot(q['cx']-sx,q['cy']-sy))
            if d<=max(W*.10,H*.12): selected.append(q)
        if not selected: selected=[seed]
        x=min(int(q['x']) for q in selected); y=min(int(q['y']) for q in selected); xx=max(int(q['x']+q['w']) for q in selected); yy=max(int(q['y']+q['h']) for q in selected)
        return Box(x,y,max(1,xx-x),max(1,yy-y)).pad(pad,W,H)

    def _text_mask(self,words,W,H,pad=8):
        mask=np.zeros((H,W),np.uint8)
        for q in words:
            b=self._box(q,pad,W,H); cv2.rectangle(mask,(b.x,b.y),(b.x+b.w,b.y+b.h),255,-1)
        return mask

    @staticmethod
    def _line_info(lines):
        out=[]
        if lines is None:return out
        for l in lines[:,0]:
            x1,y1,x2,y2=map(int,l); dx=x2-x1; dy=y2-y1; length=float(np.hypot(dx,dy))
            if length<=0:continue
            angle=float(np.degrees(np.arctan2(dy,dx)))%180
            out.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'cx':(x1+x2)/2,'cy':(y1+y2)/2,'length':length,'angle':angle})
        return out

    def _structural_lines(self,img,words):
        H,W=img.shape[:2]; edge=cv2.Canny(self._gray(img),45,150); edge[self._text_mask(words,W,H,7)>0]=0
        edge[:max(2,int(H*.035)),:]=0; edge[int(H*.965):,:]=0; edge[:,:max(2,int(W*.02))]=0; edge[:,int(W*.98):]=0
        min_len=max(35,int(min(W,H)*self.min_line_length_ratio))
        lines=cv2.HoughLinesP(edge,1,np.pi/180,threshold=max(28,int(min(W,H)*.012)),minLineLength=min_len,maxLineGap=10)
        info=self._line_info(lines); kept=[]
        for q in sorted(info,key=lambda z:z['length'],reverse=True):
            if any(abs(q['cx']-r['cx'])<8 and abs(q['cy']-r['cy'])<8 and min(abs(q['angle']-r['angle']),180-abs(q['angle']-r['angle']))<5 and abs(q['length']-r['length'])<max(12,.12*r['length']) for r in kept):continue
            kept.append(q)
        return kept[:300]

    def _line_pair(self,a,b,W,H):
        d=float(np.hypot(a['cx']-b['cx'],a['cy']-b['cy'])); da=abs(a['angle']-b['angle']); da=min(da,180-da); lr=abs(a['length']-b['length'])/max(a['length'],b['length'])
        return d<=max(W,H)*self.line_position_ratio and da<=self.line_angle_deg and lr<=self.line_length_ratio

    def _geometry_regions(self,before,after,old_words,new_words,occupied):
        H,W=before.shape[:2]; old_lines=self._structural_lines(before,old_words); new_lines=self._structural_lines(after,new_words); candidates=[]; used_new=set()
        for a in old_lines:
            matches=[]
            for ni,b in enumerate(new_lines):
                if ni in used_new:continue
                if self._line_pair(a,b,W,H): matches.append((np.hypot(a['cx']-b['cx'],a['cy']-b['cy'])+abs(a['angle']-b['angle'])*2,ni,b))
            if matches:
                matches.sort(); _,ni,b=matches[0]; used_new.add(ni); da=abs(a['angle']-b['angle']); da=min(da,180-da); lr=abs(a['length']-b['length'])/max(a['length'],b['length'])
                if da>3.0 or lr>.10:candidates.append((a,b,.82))
            else:candidates.append((a,None,.68))
        for ni,b in enumerate(new_lines):
            if ni not in used_new:candidates.append((None,b,.68))
        regions=[]
        for a,b,conf in candidates:
            pts=[]
            if a:pts += [(a['x1'],a['y1']),(a['x2'],a['y2'])]
            if b:pts += [(b['x1'],b['y1']),(b['x2'],b['y2'])]
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; x=min(xs); y=min(ys); xx=max(xs); yy=max(ys); pad=max(12,int(min(W,H)*.008)); box=Box(max(0,x-pad),max(0,y-pad),min(W,xx+pad)-max(0,x-pad),min(H,yy+pad)-max(0,y-pad))
            if box.w<=8 or box.h<=8 or box.w>W*.45 and box.h>H*.45:continue
            if any(self._iou(box,q)>.55 for q in occupied):continue
            regions.append(ChangeRegion(box.x,box.y,box.w,box.h,box.w*box.h,0.0,'geometry_change',conf,self._crop(before,box),self._crop(after,box),None,'','', 'geometry_change'))
        return regions

    def _mergeable(self,a,b):
        if a.change_kind!=b.change_kind or a.change_kind=='geometry_change':return False
        ax,ay,axx,ayy=a.x,a.y,a.x+a.width,a.y+a.height; bx,by,bxx,byy=b.x,b.y,b.x+b.width,b.y+b.height; gapx=max(0,max(ax,bx)-min(axx,bxx)); gapy=max(0,max(ay,by)-min(ayy,byy)); ox=max(0,min(axx,bxx)-max(ax,bx)); oy=max(0,min(ayy,byy)-max(ay,by)); return (gapx<=self.merge_gap and oy>0) or (gapy<=self.merge_gap and ox>0) or self._iou(Box(ax,ay,a.width,a.height),Box(bx,by,b.width,b.height))>.05

    def _merge_regions(self,regions,before,after):
        pending=list(regions); changed=True
        while changed:
            changed=False; out=[]
            while pending:
                cur=pending.pop(0); merged=False
                for i,other in enumerate(pending):
                    if not self._mergeable(cur,other):continue
                    x=min(cur.x,other.x); y=min(cur.y,other.y); xx=max(cur.x+cur.width,other.x+other.width); yy=max(cur.y+cur.height,other.y+other.height); box=Box(x,y,xx-x,yy-y); ot=' '.join(t for t in (cur.old_text,other.old_text) if t).strip(); nt=' '.join(t for t in (cur.new_text,other.new_text) if t).strip()
                    cur=ChangeRegion(x,y,xx-x,yy-y,(xx-x)*(yy-y),0.0,cur.region_type,max(cur.confidence,other.confidence),self._crop(before,box) if ot else np.full((box.h,box.w,3),255,np.uint8),self._crop(after,box) if nt else np.full((box.h,box.w,3),255,np.uint8),None,ot,nt,cur.change_kind); pending.pop(i); merged=True; changed=True; break
                if not merged:out.append(cur)
            pending=out
        return pending

    def detect(self,before_page,after_page,aligned_after=None,alignment_matrix=None):
        try:
            before=self._img(before_page); after=self._img(aligned_after if aligned_after is not None else after_page); H,W=before.shape[:2]; old=self._words(before_page); raw_new=self._words(after_page); M=alignment_matrix
            if M is not None:new=self._mapped_words(raw_new,self._img(after_page).shape,after.shape,M)
            else:
                ah,aw=self._img(after_page).shape[:2]; sx=W/max(1,aw); sy=H/max(1,ah); new=[{**q,'x':q['x']*aw*sx,'y':q['y']*ah*sy,'w':q['w']*aw*sx,'h':q['h']*ah*sy,'cx':q['cx']*aw*sx,'cy':q['cy']*ah*sy} for q in raw_new]
            oldpx=[{**q,'x':q['x']*W,'y':q['y']*H,'w':q['w']*W,'h':q['h']*H,'cx':q['cx']*W,'cy':q['cy']*H} for q in old]; pairs,used_o,used_n=self._pair_words(oldpx,new,W,H); regions=[]
            for o,n,score in pairs:
                if self._norm(o['text'])==self._norm(n['text']):continue
                cls=o['class']; ob=self._box(o,10,W,H); nb=self._box(n,10,W,H)
                if cls=='NOTE':
                    ob=self._expand_note_box(o,oldpx,W,H); nb=self._expand_note_box(n,new,W,H)
                x=min(ob.x,nb.x); y=min(ob.y,nb.y); xx=max(ob.x+ob.w,nb.x+nb.w); yy=max(ob.y+ob.h,nb.y+nb.h); b=Box(x,y,xx-x,yy-y).pad(8,W,H); conf=max(.55,min(.99,1-score))
                if cls in ('NOTE','TEXT') and not (self._value_like(o['text']) and self._value_like(n['text'])):
                    okind='note_deleted' if cls=='NOTE' else 'text_deleted'; nkind='note_added' if cls=='NOTE' else 'text_added'; self._add_region(regions,before,after,ob,okind,o['text'],'',conf); self._add_region(regions,before,after,nb,nkind,'',n['text'],conf)
                else:
                    typ={'DIMENSION':'dimension_change','GDT':'gdt_change','NOTE':'note_change','TEXT':'text_change'}[cls]; regions.append(ChangeRegion(b.x,b.y,b.w,b.h,b.w*b.h,0,typ,conf,self._crop(before,b),self._crop(after,b),None,o['text'],n['text'],typ))
            matched_boxes=[self._box(o,28,W,H) for o,_,_ in pairs]+[self._box(n,28,W,H) for _,n,_ in pairs]
            for side,words,used in [('deleted',oldpx,used_o),('added',new,used_n)]:
                for i,q in enumerate(words):
                    if i in used:continue
                    qb=self._box(q,10,W,H)
                    if not any(self._iou(qb,m)>0.02 or (abs(q['cx']-(m.x+m.w/2))<max(35,m.w) and abs(q['cy']-(m.y+m.h/2))<max(35,m.h)) for m in matched_boxes):continue
                    cls=q['class']; typ={'DIMENSION':'dimension_added' if side=='added' else 'dimension_deleted','GDT':'gdt_added' if side=='added' else 'gdt_deleted','NOTE':'note_added' if side=='added' else 'note_deleted','TEXT':'text_added' if side=='added' else 'text_deleted'}[cls]; b=qb.pad(8,W,H); self._add_region(regions,before,after,b,typ,q['text'] if side=='deleted' else '',q['text'] if side=='added' else '',.65)
            occupied=[Box(r.x,r.y,r.width,r.height) for r in regions if r.change_kind!='geometry_change']; geometry=self._geometry_regions(before,after,oldpx,new,occupied); regions.extend(geometry); before_merge=len(regions); final=self._merge_regions(regions,before,after); final2=[]
            for r in sorted(final,key=lambda z:(z.confidence,z.area),reverse=True):
                rb=Box(r.x,r.y,r.width,r.height)
                if not any(self._iou(rb,Box(q.x,q.y,q.width,q.height))>.60 for q in final2):final2.append(r)
            reason=f'v3 native={len(old)}/{len(raw_new)}, pairs={len(pairs)}, semantic={before_merge-len(geometry)}, geometry={len(geometry)}, merged={len(final)}, final={len(final2)}, unmatched={len(old)-len(used_o)}/{len(raw_new)-len(used_n)}'
            return ChangeDetectionResult(True,final2,None,None,0.0,reason)
        except Exception as exc:
            return ChangeDetectionResult(False,[],reason=f'v3_error: {type(exc).__name__}: {exc}')
