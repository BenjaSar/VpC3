#!/usr/bin/env python3
"""
MLflow Training Dashboard Generator
Pulls data from MLflow and generates an interactive HTML dashboard
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import mlflow
# import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

def generate_dashboard(output_dir='outputs/training_dashboard', experiment_name='floor-plan-segmentation'):
    """
    Generate HTML dashboard from MLflow data
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get MLflow experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found in MLflow")
        print("   Start training first to create experiment data")
        return
    
    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) == 0:
        print("❌ No training runs found")
        return
    
    # Get latest run
    latest_run = runs.iloc[0]
    run_id = latest_run['run_id']
    
    print(f"✓ Found experiment: {experiment_name}")
    print(f"✓ Latest run: {run_id}")
    print(f"✓ Status: {latest_run['status']}")
    
    # Get run data
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)
    metrics = run_data.data.metrics
    params = run_data.data.params
    
    print(f"✓ Loaded {len(metrics)} metrics")
    
    # Get history from checkpoint
    checkpoint_dir = Path('models/checkpoints')
    history_file = checkpoint_dir / 'training_history.json'
    
    history = {}
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        print(f"✓ Loaded training history ({len(history.get('train_loss', []))} epochs)")
    
    # Generate HTML dashboard
    html_content = generate_html(
        run_id=run_id,
        experiment_name=experiment_name,
        params=params,
        metrics=metrics,
        history=history,
        run_status=latest_run['status']
    )
    
    # Save HTML
    html_path = output_path / 'dashboard.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Dashboard saved to: {html_path}")
    print(f"\nOpen in browser: file://{html_path.absolute()}")


def generate_html(run_id, experiment_name, params, metrics, history, run_status):
    """Generate HTML dashboard content"""
    
    # Prepare data for charts
    epochs = list(range(1, len(history.get('train_loss', [])) + 1))
    
    # Calculate best metrics
    if history.get('val_iou'):
        best_iou = max(history['val_iou'])
        best_iou_epoch = history['val_iou'].index(best_iou) + 1
    else:
        best_iou = 0
        best_iou_epoch = 0
    
    if history.get('val_loss'):
        best_val_loss = min(history['val_loss'])
        best_val_loss_epoch = history['val_loss'].index(best_val_loss) + 1
    else:
        best_val_loss = 0
        best_val_loss_epoch = 0
    
    current_epoch = len(history.get('train_loss', []))
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Floor Plan Segmentation - Training Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                padding: 20px;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .header h1 {{
                color: #667eea;
                margin-bottom: 10px;
            }}
            .header p {{
                color: #666;
                margin: 5px 0;
            }}
            .status {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                margin-left: 10px;
            }}
            .status.running {{
                background: #4CAF50;
                color: white;
            }}
            .status.finished {{
                background: #2196F3;
                color: white;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metric-card h3 {{
                color: #667eea;
                font-size: 14px;
                text-transform: uppercase;
                margin-bottom: 10px;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }}
            .metric-subtext {{
                font-size: 12px;
                color: #999;
                margin-top: 5px;
            }}
            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .chart-container {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .chart-container h3 {{
                color: #667eea;
                margin-bottom: 15px;
            }}
            .params-table {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .params-table h3 {{
                color: #667eea;
                margin-bottom: 15px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }}
            th {{
                background: #f5f5f5;
                font-weight: bold;
                color: #667eea;
            }}
            .footer {{
                text-align: center;
                color: white;
                margin-top: 20px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Floor Plan Segmentation - Training Dashboard
                    <span class="status {'running' if run_status == 'RUNNING' else 'finished'}">
                        {run_status}
                    </span>
                </h1>
                <p><strong>Experiment:</strong> {experiment_name}</p>
                <p><strong>Run ID:</strong> {run_id}</p>
                <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Current Epoch:</strong> {current_epoch}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Best Val IoU</h3>
                    <div class="metric-value">{best_iou:.4f}</div>
                    <div class="metric-subtext">Epoch {best_iou_epoch}</div>
                </div>
                <div class="metric-card">
                    <h3>Best Val Loss</h3>
                    <div class="metric-value">{best_val_loss:.4f}</div>
                    <div class="metric-subtext">Epoch {best_val_loss_epoch}</div>
                </div>
                <div class="metric-card">
                    <h3>Current Epoch</h3>
                    <div class="metric-value">{current_epoch}</div>
                    <div class="metric-subtext">of {params.get('num_epochs', '?')} total</div>
                </div>
                <div class="metric-card">
                    <h3>Learning Rate</h3>
                    <div class="metric-value">{params.get('learning_rate', 'N/A')}</div>
                    <div class="metric-subtext">Optimized LR</div>
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Loss Curves</h3>
                    <canvas id="lossChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>IoU Curves</h3>
                    <canvas id="iouChart"></canvas>
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Active Classes</h3>
                    <canvas id="classesChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Learning Rate Schedule</h3>
                    <canvas id="lrChart"></canvas>
                </div>
            </div>
            
            <div class="params-table">
                <h3>Configuration Parameters</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add parameters
    for key, value in sorted(params.items()):
        html += f"<tr><td>{key}</td><td>{value}</td></tr>"
    
    html += """
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Floor Plan Segmentation Model - Training Dashboard</p>
                <p>Generated with MLflow & PyTorch</p>
            </div>
        </div>
        
        <script>
            // Prepare data
            const epochs = """ + json.dumps(epochs) + """;
            const trainLoss = """ + json.dumps(history.get('train_loss', [])) + """;
            const valLoss = """ + json.dumps(history.get('val_loss', [])) + """;
            const trainIoU = """ + json.dumps(history.get('train_iou', [])) + """;
            const valIoU = """ + json.dumps(history.get('val_iou', [])) + """;
            const activeClasses = """ + json.dumps(history.get('active_classes', [])) + """;
            const lrSchedule = """ + json.dumps(history.get('lr', [])) + """;
            
            // Loss Chart
            new Chart(document.getElementById('lossChart'), {
                type: 'line',
                data: {
                    labels: epochs,
                    datasets: [
                        {
                            label: 'Train Loss',
                            data: trainLoss,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Val Loss',
                            data: valLoss,
                            borderColor: '#764ba2',
                            backgroundColor: 'rgba(118, 75, 162, 0.1)',
                            tension: 0.4,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: true
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // IoU Chart
            new Chart(document.getElementById('iouChart'), {
                type: 'line',
                data: {
                    labels: epochs,
                    datasets: [
                        {
                            label: 'Train IoU',
                            data: trainIoU,
                            borderColor: '#4CAF50',
                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Val IoU',
                            data: valIoU,
                            borderColor: '#FF9800',
                            backgroundColor: 'rgba(255, 152, 0, 0.1)',
                            tension: 0.4,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: true
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
            
            // Active Classes Chart
            new Chart(document.getElementById('classesChart'), {
                type: 'line',
                data: {
                    labels: epochs,
                    datasets: [
                        {
                            label: 'Active Classes',
                            data: activeClasses,
                            borderColor: '#9C27B0',
                            backgroundColor: 'rgba(156, 39, 176, 0.1)',
                            tension: 0.4,
                            fill: true,
                            pointRadius: 3,
                            pointBackgroundColor: '#9C27B0'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: true
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 12
                        }
                    }
                }
            });
            
            // Learning Rate Chart
            new Chart(document.getElementById('lrChart'), {
                type: 'line',
                data: {
                    labels: epochs,
                    datasets: [
                        {
                            label: 'Learning Rate',
                            data: lrSchedule,
                            borderColor: '#FF6B6B',
                            backgroundColor: 'rgba(255, 107, 107, 0.1)',
                            tension: 0.4,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: true
                        }
                    },
                    scales: {
                        y: {
                            type: 'logarithmic'
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    
    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MLflow training dashboard')
    parser.add_argument('--output', type=str, default='outputs/training_dashboard',
                       help='Output directory for dashboard')
    parser.add_argument('--experiment', type=str, default='floor-plan-segmentation',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    generate_dashboard(output_dir=args.output, experiment_name=args.experiment)
