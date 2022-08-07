from pdf2image import convert_from_path
import time

jpegopt={
    "quality": 100,
    "progressive": True,
    "optimize": True
}

time_str = str(int(round(time.time() * 1000)))

print("Loading PDF file...")
convert_from_path("shamaryati code 259200305.pdf",
    dpi=320,
    output_folder="pdf_img/",
    first_page=3,
    last_page=56,
    fmt="jpeg",
    jpegopt=jpegopt,
    thread_count=5,
    userpw=None,
    use_cropbox=False,
    strict=False,
    transparent=False,
    single_file=False,
    output_file=time_str,
    poppler_path=r'C:\Program Files\poppler-0.68.0\bin',
    grayscale=False,
    size=None,
    paths_only=False,
    hide_annotations=False,)

print("Done!")