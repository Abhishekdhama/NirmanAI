import os
import datetime
from jinja2 import Environment, FileSystemLoader

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

def generate_report_data(project_name, project_type, state, current_month, models=None):
    """
    Generates data for the report. Uses mock data if models=None or simulation fails.
    """
    mock_data = {
        'executive_summary': f"This report provides an overview of supply chain risks for the {project_name} project in {state} during {current_month}. Overall risk remains moderate with some material shortages expected.",
        'weather': {
            'condition': 'Heavy Rain',
            'impact': 'Possible delays in aggregate transport and concrete curing.'
        },
        'risk_deliveries': [
            {'material': 'Cement', 'supplier': 'UltraTech', 'risk_level': 'High', 'risk_level_color': 'red', 'reason': 'Local transport strike'},
            {'material': 'Steel Rebar', 'supplier': 'Tata Steel', 'risk_level': 'Medium', 'risk_level_color': 'orange', 'reason': 'Price volatility, slight delay in dispatch'},
            {'material': 'River Sand', 'supplier': 'Local Suppliers', 'risk_level': 'High', 'risk_level_color': 'red', 'reason': 'Monsoon mining restrictions'},
            {'material': 'Bricks', 'supplier': 'Regional Kilns', 'risk_level': 'Low', 'risk_level_color': 'green', 'reason': 'Adequate stock available'},
            {'material': 'Aggregate', 'supplier': 'Quarry A', 'risk_level': 'Medium', 'risk_level_color': 'yellow', 'reason': 'Rain-induced loading delays'}
        ],
        'wastage_forecasts': [
            {'material': 'Cement', 'forecast': '6.5', 'threshold': '5.0', 'status': 'Exceeded', 'status_color': 'red'},
            {'material': 'Steel', 'forecast': '2.0', 'threshold': '3.0', 'status': 'Normal', 'status_color': 'green'},
            {'material': 'Tiles', 'forecast': '8.0', 'threshold': '6.0', 'status': 'Exceeded', 'status_color': 'orange'}
        ],
        'recommended_actions': [
            "Stockpile cement before the upcoming transport strike.",
            "Diversify sand suppliers to mitigate monsoon-related shortages.",
            "Implement stricter material handling for tiles to reduce breakage wastage.",
            "Schedule steel rebar deliveries during clear weather windows."
        ]
    }
    
    if models is not None:
        try:
            from simulation_engine import run_simulation, optimize_procurement
            try:
                from weather import get_weather_risk_summary
            except ImportError:
                get_weather_risk_summary = None
            
            # Map month to int if needed
            if isinstance(current_month, int):
                month_int = current_month
            else:
                months = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
                          "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
                month_int = months.get(str(current_month), 7)
            
            sim_result = run_simulation(project_type, state, month_int, n_simulations=1000, models=models)
            summary = sim_result["executive_summary"]
            
            # Weather risk
            weather_data = mock_data["weather"]
            if get_weather_risk_summary:
                try:
                    weather_risk = get_weather_risk_summary(state)
                    if weather_risk:
                        weather_data = {
                            "condition": weather_risk.get("description", "Unknown"),
                            "impact": f"Delay impact: +{weather_risk.get('impact_on_delivery', 0)}%"
                        }
                except Exception as e:
                    print(f"Weather error: {str(e)}")
            
            # Risk Deliveries
            risk_deliveries = []
            for m in sim_result.get("material_risk_ranking", []):
                freq = m["delay_frequency"]
                level = "High" if freq > 50 else ("Medium" if freq > 20 else "Low")
                color = "red" if level == "High" else ("orange" if level == "Medium" else "green")
                risk_deliveries.append({
                    "material": m["material"],
                    "supplier": "TBD",
                    "risk_level": level,
                    "risk_level_color": color,
                    "reason": f"Delayed in {freq}% of simulations"
                })
                
            if not risk_deliveries:
                risk_deliveries = mock_data['risk_deliveries']
                
            actions = summary.get("recommended_actions", mock_data['recommended_actions'])
            if not actions:
                actions = mock_data['recommended_actions']
                
            return {
                'executive_summary': summary.get("headline", mock_data['executive_summary']),
                'weather': weather_data,
                'risk_deliveries': risk_deliveries,
                'wastage_forecasts': mock_data['wastage_forecasts'],
                'recommended_actions': actions
            }
        except Exception as e:
            # Ensure ASCII only in print
            err_msg = str(e).encode("ascii", errors="ignore").decode("ascii")
            print(f"Failed to use simulation engine, using mock data. Error: {err_msg}")
            return mock_data
            
    return mock_data

def generate_pdf_report(report_id, project_name, project_type, state, current_month, output_dir='reports/generated', models=None):
    """
    Generates a PDF report using Jinja2 and Weasyprint.
    Falls back to HTML if Weasyprint is not available.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Jinja2 environment
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('report.html')
    
    # Get report data
    data = generate_report_data(project_name, project_type, state, current_month, models)
    
    # Add common context
    context = {
        'report_id': report_id,
        'project_name': project_name,
        'project_type': project_type,
        'state': state,
        'current_month': current_month,
        'generation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    context.update(data)
    
    # Render HTML
    rendered_html = template.render(context)
    
    if WEASYPRINT_AVAILABLE:
        output_filepath = os.path.join(output_dir, f"{report_id}.pdf")
        try:
            HTML(string=rendered_html, base_url=__file__).write_pdf(output_filepath)
            print(f"Successfully generated PDF report: {output_filepath}")
            return output_filepath
        except Exception as e:
            print(f"Error generating PDF with WeasyPrint: {e}")
            print("Falling back to HTML format.")
    else:
        print("WeasyPrint is not installed. Falling back to HTML format.")
    
    # Fallback to HTML
    output_filepath = os.path.join(output_dir, f"{report_id}.html")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(rendered_html)
    
    print(f"Successfully generated HTML report: {output_filepath}")
    return output_filepath

if __name__ == "__main__":
    report_file = generate_pdf_report(
        report_id="RPT-2026-001",
        project_name="Metro Line 3 Phase 1",
        project_type="Infrastructure",
        state="Maharashtra",
        current_month="July",
        output_dir=os.path.join(os.path.dirname(__file__), "reports", "generated")
    )
    print(f"Report generation completed. File saved at: {report_file}")
