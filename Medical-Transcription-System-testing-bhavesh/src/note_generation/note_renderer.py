from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import json
import re
import html
import markdown
import traceback
from jinja2 import Template

class NoteRenderer:
    """
    Class for rendering clinical notes in different formats (HTML, PDF)
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the renderer with templates
        
        Args:
            templates_dir: Directory containing templates (optional)
        """
        if templates_dir and os.path.isdir(templates_dir):
            self.templates_dir = templates_dir
        else:
            # Default to templates directory in the same folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.templates_dir = os.path.join(current_dir, "templates")
            
            # Create templates directory if it doesn't exist
            if not os.path.exists(self.templates_dir):
                os.makedirs(self.templates_dir)
        
        # Default SOAP template
        self.default_soap_template = """<!DOCTYPE html>
<html>
<head>
    <title>SOAP Note</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        h1 {
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
        }
        h3 {
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .alert {
            background-color: #fff4e5;
            border-left: 4px solid #ff9800;
            padding: 10px;
            margin: 20px 0;
        }
        .alert.high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
        }
        .medication {
            margin-bottom: 8px;
        }
        .date {
            text-align: right;
            margin-bottom: 20px;
            font-style: italic;
        }
        .signature {
            margin-top: 60px;
            border-top: 1px solid #999;
            padding-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Clinical SOAP Note</h1>
    <div class="date">Date: {{date}}</div>
    
    <h2>S: Subjective</h2>
    {% if subjective.chief_complaint %}
    <h3>Chief Complaint</h3>
    <p>{{subjective.chief_complaint}}</p>
    {% endif %}
    
    {% if subjective.history_of_present_illness %}
    <h3>History of Present Illness</h3>
    <p>{{subjective.history_of_present_illness}}</p>
    {% endif %}
    
    {% if subjective.current_medications %}
    <h3>Current Medications</h3>
    <p>{{subjective.current_medications}}</p>
    {% endif %}
    
    <h2>O: Objective</h2>
    {% if objective.vitals %}
    <h3>Vitals</h3>
    <p>
        {% if objective.vitals.temperature %}Temperature: {{objective.vitals.temperature}}<br>{% endif %}
        {% if objective.vitals.heart_rate %}Heart Rate: {{objective.vitals.heart_rate}}<br>{% endif %}
        {% if objective.vitals.blood_pressure %}Blood Pressure: {{objective.vitals.blood_pressure}}<br>{% endif %}
        {% if objective.vitals.respiratory_rate %}Respiratory Rate: {{objective.vitals.respiratory_rate}}<br>{% endif %}
        {% if objective.vitals.oxygen_saturation %}O2 Saturation: {{objective.vitals.oxygen_saturation}}{% endif %}
    </p>
    {% endif %}
    
    {% if objective.physical_exam %}
    <h3>Physical Examination</h3>
    <p>{{objective.physical_exam}}</p>
    {% endif %}
    
    <h2>A: Assessment</h2>
    {% if assessment.diagnosis %}
    <p>{{assessment.diagnosis}}</p>
    {% endif %}
    
    <h2>P: Plan</h2>
    {% if plan.current_medications %}
    <h3>Current Medications</h3>
    {% for med in plan.current_medications %}
    <div class="medication">• {{med.name}} {{med.dose}} {{med.frequency}}</div>
    {% endfor %}
    {% endif %}
    
    {% if plan.new_prescriptions %}
    <h3>New Prescriptions</h3>
    {% for med in plan.new_prescriptions %}
    <div class="medication">• {{med.name}} {{med.dose}} {{med.frequency}}</div>
    {% endfor %}
    {% endif %}
    
    {% if plan.plan_text %}
    <h3>Additional Plan</h3>
    <p>{{plan.plan_text}}</p>
    {% endif %}
    
    {% if alerts %}
    <h2>Alerts</h2>
    {% for alert in alerts %}
    <div class="alert {% if alert.severity == 'high' %}high{% endif %}">
        <strong>{{alert.severity|capitalize}} Alert:</strong> {{alert.description}}
    </div>
    {% endfor %}
    {% endif %}
    
    <div class="signature">
        <p>Provider Signature: ______________________________ Date: ___________</p>
    </div>
</body>
</html>"""

    # Add this at the top with your other imports:
    def check_pdf_dependencies():
        """Check for PDF generation dependencies"""
        try:
            import pdfkit
            return True
        except ImportError:
            print("Warning: pdfkit not found. PDF generation will not be available.")
            print("To enable PDF generation, install pdfkit and wkhtmltopdf:")
            print("  pip install pdfkit")
            print("  Download wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
            return False

    # Then add this method to the NoteRenderer class:
    def render_pdf(self, note_data: Dict[str, Any], output_path: str) -> bool:
        """
        Render the note as PDF and save to file
        
        Args:
            note_data: The note data to render
            output_path: Path to save the PDF
            
        Returns:
            True if PDF was successfully created, False otherwise
        """
        try:
            import pdfkit
        except ImportError:
            print("Error: pdfkit not found. Cannot generate PDF.")
            print("To enable PDF generation, install pdfkit and wkhtmltopdf:")
            print("  pip install pdfkit")
            print("  Download wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
            return False
            
        # First render to HTML
        html_content = self.render_html(note_data)
        
        # Path to wkhtmltopdf - try to auto-detect on Windows
        wkhtmltopdf_path = None
        if os.name == 'nt':  # Windows
            possible_paths = [
                r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
                r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe',
                # Add the path where you installed wkhtmltopdf if it's different
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    wkhtmltopdf_path = path
                    print(f"Found wkhtmltopdf at: {wkhtmltopdf_path}")
                    break
        
        if not wkhtmltopdf_path:
            print("Warning: wkhtmltopdf path not found. Make sure it's installed.")
            print("You can download it from: https://wkhtmltopdf.org/downloads.html")
        
        # Configure PDF options
        options = {
            'page-size': 'Letter',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
            'quiet': '',
        }
        
        config = None
        if wkhtmltopdf_path:
            config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Generate PDF
            if config:
                pdfkit.from_string(html_content, output_path, options=options, configuration=config)
            else:
                pdfkit.from_string(html_content, output_path, options=options)
                
            if os.path.exists(output_path):
                print(f"PDF generated successfully: {output_path}")
                return True
            else:
                print(f"Error: PDF file was not created at {output_path}")
                return False
        except Exception as e:
            print(f"Error generating PDF: {e}")
            traceback.print_exc()  # Now traceback is imported so this will work
            return False

    def render(self, note_data: Dict[str, Any], output_format: str = "txt") -> str:
        """
        Render the note in the specified format
        
        Args:
            note_data: The note data to render
            output_format: The output format ("txt", "html", "md")
            
        Returns:
            The rendered note as a string
        """
        if output_format == "html":
            return self.render_html(note_data)
        elif output_format == "md":
            return self.render_markdown(note_data)
        else:
            return self.render_text(note_data)
    
    def render_html(self, note_data: Dict[str, Any]) -> str:
        """
        Render the note as HTML
        
        Args:
            note_data: The note data to render
            
        Returns:
            HTML string
        """
        # Add date if not present
        if "date" not in note_data:
            note_data["date"] = datetime.now().strftime("%Y-%m-%d")
            
        # Sanitize data to prevent injection issues
        sanitized_data = self._sanitize_data(note_data)
            
        # Try to load the SOAP template from file
        template_path = os.path.join(self.templates_dir, "soap_template.html")
        if os.path.isfile(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                template_str = f.read()
        else:
            # Use default template
            template_str = self.default_soap_template
            
            # Save default template for future use
            try:
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template_str)
            except:
                pass
                
        # Render template
        template = Template(template_str)
        html_content = template.render(**sanitized_data)
        
        return html_content
    
    def render_text(self, note_data: Dict[str, Any]) -> str:
        """
        Render the note as plain text
        
        Args:
            note_data: The note data to render
            
        Returns:
            Text string
        """
        text_lines = []
        
        # Add header
        text_lines.append("CLINICAL SOAP NOTE")
        text_lines.append("-" * 80)
        
        # Add date
        date = note_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        text_lines.append(f"Date: {date}")
        text_lines.append("")
        
        # Subjective section
        text_lines.append("S: SUBJECTIVE")
        text_lines.append("-" * 80)
        
        subjective = note_data.get("subjective", {})
        if subjective.get("chief_complaint"):
            text_lines.append("Chief Complaint:")
            text_lines.append(subjective["chief_complaint"])
            text_lines.append("")
            
        if subjective.get("history_of_present_illness"):
            text_lines.append("History of Present Illness:")
            text_lines.append(subjective["history_of_present_illness"])
            text_lines.append("")
            
        if subjective.get("current_medications"):
            text_lines.append("Current Medications:")
            text_lines.append(subjective["current_medications"])
            text_lines.append("")
        
        # Objective section
        text_lines.append("O: OBJECTIVE")
        text_lines.append("-" * 80)
        
        objective = note_data.get("objective", {})
        if objective.get("vitals"):
            text_lines.append("Vitals:")
            vitals = objective["vitals"]
            if vitals.get("temperature"):
                text_lines.append(f"Temperature: {vitals['temperature']}")
            if vitals.get("heart_rate"):
                text_lines.append(f"Heart Rate: {vitals['heart_rate']}")
            if vitals.get("blood_pressure"):
                text_lines.append(f"Blood Pressure: {vitals['blood_pressure']}")
            if vitals.get("respiratory_rate"):
                text_lines.append(f"Respiratory Rate: {vitals['respiratory_rate']}")
            if vitals.get("oxygen_saturation"):
                text_lines.append(f"O2 Saturation: {vitals['oxygen_saturation']}")
            text_lines.append("")
            
        if objective.get("physical_exam"):
            text_lines.append("Physical Examination:")
            text_lines.append(objective["physical_exam"])
            text_lines.append("")
        
        # Assessment section
        text_lines.append("A: ASSESSMENT")
        text_lines.append("-" * 80)
        
        assessment = note_data.get("assessment", {})
        if assessment.get("diagnosis"):
            text_lines.append(assessment["diagnosis"])
            text_lines.append("")
        
        # Plan section
        text_lines.append("P: PLAN")
        text_lines.append("-" * 80)
        
        plan = note_data.get("plan", {})
        
        # Display current medications
        if plan.get("current_medications"):
            text_lines.append("Current Medications:")
            for med in plan["current_medications"]:
                med_line = f"• {med.get('name', '')} {med.get('dose', '')} {med.get('frequency', '')}"
                text_lines.append(med_line)
            text_lines.append("")
        
        # Display new prescriptions
        if plan.get("new_prescriptions"):
            text_lines.append("New Prescriptions:")
            for med in plan["new_prescriptions"]:
                med_line = f"• {med.get('name', '')} {med.get('dose', '')} {med.get('frequency', '')}"
                text_lines.append(med_line)
            text_lines.append("")
        
        if plan.get("plan_text"):
            text_lines.append("Additional Plan:")
            text_lines.append(plan["plan_text"])
            text_lines.append("")
        
        # Alerts section
        alerts = note_data.get("alerts", [])
        if alerts:
            text_lines.append("ALERTS")
            text_lines.append("-" * 80)
            for alert in alerts:
                severity = alert.get("severity", "").capitalize()
                description = alert.get("description", "")
                text_lines.append(f"{severity} Alert: {description}")
            text_lines.append("")
        
        # Signature
        text_lines.append("-" * 80)
        text_lines.append("Provider Signature: ____________________________ Date: ___________")
        
        return "\n".join(text_lines)
    
    def render_markdown(self, note_data: Dict[str, Any]) -> str:
        """
        Render the note as Markdown
        
        Args:
            note_data: The note data to render
            
        Returns:
            Markdown string
        """
        md_lines = []
        
        # Add header
        md_lines.append("# CLINICAL SOAP NOTE")
        
        # Add date
        date = note_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        md_lines.append(f"**Date:** {date}")
        md_lines.append("")
        
        # Subjective section
        md_lines.append("## S: SUBJECTIVE")
        
        subjective = note_data.get("subjective", {})
        if subjective.get("chief_complaint"):
            md_lines.append("### Chief Complaint")
            md_lines.append(subjective["chief_complaint"])
            md_lines.append("")
            
        if subjective.get("history_of_present_illness"):
            md_lines.append("### History of Present Illness")
            md_lines.append(subjective["history_of_present_illness"])
            md_lines.append("")
            
        if subjective.get("current_medications"):
            md_lines.append("### Current Medications")
            md_lines.append(subjective["current_medications"])
            md_lines.append("")
        
        # Objective section
        md_lines.append("## O: OBJECTIVE")
        
        objective = note_data.get("objective", {})
        if objective.get("vitals"):
            md_lines.append("### Vitals")
            vitals = objective["vitals"]
            if vitals.get("temperature"):
                md_lines.append(f"- **Temperature:** {vitals['temperature']}")
            if vitals.get("heart_rate"):
                md_lines.append(f"- **Heart Rate:** {vitals['heart_rate']}")
            if vitals.get("blood_pressure"):
                md_lines.append(f"- **Blood Pressure:** {vitals['blood_pressure']}")
            if vitals.get("respiratory_rate"):
                md_lines.append(f"- **Respiratory Rate:** {vitals['respiratory_rate']}")
            if vitals.get("oxygen_saturation"):
                md_lines.append(f"- **O2 Saturation:** {vitals['oxygen_saturation']}")
            md_lines.append("")
            
        if objective.get("physical_exam"):
            md_lines.append("### Physical Examination")
            md_lines.append(objective["physical_exam"])
            md_lines.append("")
        
        # Assessment section
        md_lines.append("## A: ASSESSMENT")
        
        assessment = note_data.get("assessment", {})
        if assessment.get("diagnosis"):
            md_lines.append(assessment["diagnosis"])
            md_lines.append("")
        
        # Plan section
        md_lines.append("## P: PLAN")
        
        plan = note_data.get("plan", {})
        
        # Display current medications
        if plan.get("current_medications"):
            md_lines.append("### Current Medications")
            for med in plan["current_medications"]:
                med_line = f"- **{med.get('name', '')}** {med.get('dose', '')} {med.get('frequency', '')}"
                md_lines.append(med_line)
            md_lines.append("")
        
        # Display new prescriptions
        if plan.get("new_prescriptions"):
            md_lines.append("### New Prescriptions")
            for med in plan["new_prescriptions"]:
                med_line = f"- **{med.get('name', '')}** {med.get('dose', '')} {med.get('frequency', '')}"
                md_lines.append(med_line)
            md_lines.append("")
        
        if plan.get("plan_text"):
            md_lines.append("### Additional Plan")
            md_lines.append(plan["plan_text"])
            md_lines.append("")
        
        # Alerts section
        alerts = note_data.get("alerts", [])
        if alerts:
            md_lines.append("## ALERTS")
            for alert in alerts:
                severity = alert.get("severity", "").capitalize()
                description = alert.get("description", "")
                md_lines.append(f"**{severity} Alert:** {description}")
            md_lines.append("")
        
        # Signature
        md_lines.append("---")
        md_lines.append("**Provider Signature:** ____________________________ **Date:** ___________")
        
        return "\n".join(md_lines)
    
    def _sanitize_data(self, data: Any) -> Any:
        """
        Sanitize data to prevent HTML injection
        
        Args:
            data: The data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            return html.escape(data)
        elif isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data