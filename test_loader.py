from core.image_loader import PDFImageLoader
 
loader = PDFImageLoader(dpi=300)
 
images = loader.load("input/before/sample_before.pdf")
 
print(f"페이지 수 : {len(images)}")
 
loader.save(images[0], "output/page1.png")
 
print("완료")