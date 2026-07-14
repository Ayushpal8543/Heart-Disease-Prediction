from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os


def generate_pdf(patient, probability, level, recommendations):

    os.makedirs("outputs", exist_ok=True)

    pdf = SimpleDocTemplate("outputs/patient_report.pdf")

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>Heart Disease Prediction Report</b>", styles["Title"]))

    story.append(Paragraph(f"<b>Risk Level:</b> {level}", styles["Normal"]))
    story.append(Paragraph(f"<b>Risk Probability:</b> {probability*100:.2f}%", styles["Normal"]))

    story.append(Paragraph("<br/><b>Patient Details</b>", styles["Heading2"]))
    

    for col in patient.columns:
        story.append(
            Paragraph(f"{col}: {patient.iloc[0][col]}", styles["Normal"])
        )

    story.append(Paragraph("<br/><b>Recommendations</b>", styles["Heading2"]))

    for rec in recommendations:
        story.append(
            Paragraph(f"• {rec}", styles["Normal"])
        )

    pdf.build(story)

    print("✔ PDF Generated")