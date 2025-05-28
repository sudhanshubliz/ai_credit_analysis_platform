from fpdf import FPDF

def generate_pdf_report(data):
    """
    Generate a PDF report for the credit risk analysis.
    Args:
        data (dict): Data to include in the report.
    Returns:
        str: Path to the generated PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Credit Risk Analysis Report", ln=True, align="C")

    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf_path = "credit_risk_report.pdf"
    pdf.output(pdf_path)
    return pdf_path