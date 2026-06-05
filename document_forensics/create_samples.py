import os
from PIL import Image, ImageDraw, ImageFont

def generate_genuine_image(output_path: str):
    """
    Generates a mock genuine bank statement image.
    
    Creates a clean, digital document canvas with header and balance info
    and saves it with no editing metadata.
    
    Parameters:
        output_path (str): Filepath to save the generated image.
    """
    # 1. Create a blank white canvas (600 width, 300 height)
    img = Image.new("RGB", (600, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 2. Draw a dark blue header bar representing the bank's branding
    draw.rectangle([(0, 0), (600, 60)], fill=(12, 35, 64))
    
    # 3. Draw bank title text
    # (Using Pillow's default font to ensure it runs out-of-the-box on any system without font files)
    draw.text((20, 18), "NATIONAL COOPERATIVE BANK", fill=(255, 255, 255))
    draw.text((450, 22), "STATEMENT OF ACCOUNT", fill=(200, 220, 240))
    
    # 4. Draw statement body text (Account Holder info, Date, and Balance)
    draw.text((40, 100), "Account Holder : JOHN D. DOE", fill=(30, 30, 30))
    draw.text((40, 125), "Account Number : 9876-5432-10", fill=(30, 30, 30))
    draw.text((40, 150), "Statement Date : June 01, 2026", fill=(30, 30, 30))
    
    # Draw a line separator
    draw.line([(40, 180), (560, 180)], fill=(200, 200, 200), width=1)
    
    # Draw the crucial financial details (the account balance)
    draw.text((40, 200), "Opening Balance: $10,500.00", fill=(100, 100, 100))
    draw.text((40, 225), "Closing Balance: $10,500.00", fill=(12, 35, 64)) # Key target for tampering
    draw.text((40, 250), "Account Status : ACTIVE / CLEAN", fill=(40, 120, 40))
    
    # 5. Inject standard scanner EXIF tags representing an untampered scan
    exif = img.getexif()
    exif[305] = "HP ScanJet Enterprise 8500"  # Software
    exif[315] = "Bank Scan Room A"            # Artist
    exif[306] = "2026:06:01 09:00:00"          # DateTime (Modification)
    exif[36867] = "2026:06:01 09:00:00"        # DateTimeOriginal (Creation)
    
    # 6. Save the image as a high-quality JPEG
    # We save with quality=98 to establish a baseline compression level
    img.save(output_path, "JPEG", quality=98, exif=exif)
    print(f"[Sample Maker] Created genuine image: {output_path}")

def generate_tampered_image(genuine_path: str, output_path: str):
    """
    Loads the genuine image, alters the account balance, and saves it with Photoshop EXIF tags.
    
    To simulate a realistic graphic forgery where an edited region has a different 
    compression history than the rest of the image, we:
    1. Resave the genuine image at a lower JPEG quality (75) to establish background compression.
    2. Load that compressed image.
    3. Modify a localized area (replace balance text).
    4. Save the final image at a higher quality (90).
    When ELA runs, the edited text (compressed once at 90) will show a high error level,
    while the background (compressed at 75 then 90) will show a low error level.
    """
    # 1. Establish background compression by saving genuine image at quality 75
    temp_background_path = genuine_path.replace(".jpg", "_temp_bg.jpg")
    with Image.open(genuine_path) as original_img:
        original_img.save(temp_background_path, "JPEG", quality=75)
        
    # 2. Open this compressed background image
    img = Image.open(temp_background_path)
    draw = ImageDraw.Draw(img)
    
    # 3. Tamper the image: draw a white box over the genuine closing balance value ($10,500.00)
    # The balance text is at (210, 225) approximately. We cover it with a white rectangle.
    draw.rectangle([(200, 220), (320, 242)], fill=(255, 255, 255))
    
    # 4. Write the fake, inflated balance over the white box
    draw.text((205, 224), "$99,999.00", fill=(12, 35, 64))
    
    # 5. Inject suspicious EXIF metadata tags to simulate Photoshop editing
    exif = img.getexif()
    exif[305] = "Adobe Photoshop 2025 (Windows)"
    exif[315] = "John Doe (Altered)"
    exif[306] = "2026:06:03 20:15:10" # Modification date
    exif[36867] = "2026:06:01 09:00:00" # Creation date
    
    # 6. Save the final tampered image at quality 90
    img.save(output_path, "JPEG", quality=90, exif=exif)
    
    # Clean up background temp file
    if os.path.exists(temp_background_path):
        os.remove(temp_background_path)
        
    print(f"[Sample Maker] Created tampered image: {output_path}")

def generate_sample_pdf(output_path: str):
    """
    Generates a simple, lightweight PDF file from raw bytes, embedding specific metadata fields.
    
    Using raw bytes allows creating a PDF structure out-of-the-box without needing external fpdf/reportlab.
    It embeds a creator ('Canva') and modification dates to test metadata detection.
    
    Parameters:
        output_path (str): Filepath to write the PDF file.
    """
    # Define a clean, minimal PDF structure in raw bytes
    # Object 1: Catalog
    # Object 2: Pages list
    # Object 3: Single Page details
    # Object 4: Page Content (stream containing text "INVOICE #987654 - Total Due: $15,000.00")
    # Object 5: Info metadata dictionary (Contains Creator=Canva, dates, author)
    pdf_bytes = (
        b"%PDF-1.4\n"
        # 1 0 obj: Root Catalog pointing to the Page tree (object 2)
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        
        # 2 0 obj: Page Tree containing 1 page (object 3)
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        
        # 3 0 obj: The single page size (Letter/A4 approx) and content stream link (object 4)
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R >>\nendobj\n"
        
        # 4 0 obj: The content stream. BT/ET encapsulates text rendering.
        b"4 0 obj\n<< /Length 75 >>\n"
        b"stream\n"
        b"BT\n"
        b"/F1 18 Tf\n"
        b"50 750 Td\n"
        b"(MOCK BANK INVOICE - LOAN APPLICATION PROOF) Tj\n"
        b"0 -40 Td\n"
        b"(Amount Disbursed: $15,000.00) Tj\n"
        b"ET\n"
        b"endstream\n"
        b"endobj\n"
        
        # 5 0 obj: The Metadata dictionary. 
        # CreationDate and ModDate are set with a 2-day gap, and Creator is marked as Canva
        b"5 0 obj\n"
        b"<< /Title (Bank Invoice Proof) "
        b"/Author (Applicant Upload) "
        b"/Creator (Canva Web Editor) "
        b"/Producer (Canva PDF Export Engine) "
        b"/CreationDate (D:20260601090000+05'30') "
        b"/ModDate (D:20260603174512+05'30') >>\n"
        b"endobj\n"
        
        # Cross-reference table (simple reconstruction)
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f\n"
        b"0000000009 00000 n\n"
        b"0000000056 00000 n\n"
        b"0000000111 00000 n\n"
        b"0000000212 00000 n\n"
        b"0000000350 00000 n\n"
        
        # Trailer pointing to Catalog and Info dictionaries
        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R /Info 5 0 R >>\n"
        b"startxref\n"
        b"505\n"
        b"%%EOF\n"
    )
    
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
        
    print(f"[Sample Maker] Created PDF file: {output_path}")

def main():
    print("=" * 60)
    print("        GENERATING MOCK TEST SAMPLES FOR FORENSICS        ")
    print("=" * 60)
    
    # Define directories
    target_dir = "./test_files"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    genuine_img_path = os.path.join(target_dir, "sample_genuine.jpg")
    tampered_img_path = os.path.join(target_dir, "sample_tampered.jpg")
    invoice_pdf_path = os.path.join(target_dir, "sample_invoice.pdf")
    
    # Create the files
    generate_genuine_image(genuine_img_path)
    generate_tampered_image(genuine_img_path, tampered_img_path)
    generate_sample_pdf(invoice_pdf_path)
    
    print("=" * 60)
    print("All sample files generated successfully under './test_files/'!")
    print("You can now run 'python run_pipeline.py -i <file_path>' to analyze them.")
    print("=" * 60)

if __name__ == "__main__":
    main()
