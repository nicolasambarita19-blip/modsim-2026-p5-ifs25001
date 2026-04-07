import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Pembangunan Gedung FITE 5 Lantai",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stage-card {
        background-color: #F8FAFC;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 3px solid #F59E0B;
    }
    .warning-box {
        background-color: #FFF3CD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .risk-high { color: #DC2626; font-weight: bold; }
    .risk-med  { color: #D97706; font-weight: bold; }
    .risk-low  { color: #16A34A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. KELAS PEMODELAN SISTEM (diadaptasi untuk konstruksi gedung)
# ============================================================================
class ConstructionPhase:
    """Kelas untuk memodelkan tahapan konstruksi gedung dengan faktor risiko"""
    
    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []
        
    def sample_duration(self, n_simulations,
                        weather_risk_mult=1.0,
                        material_risk_mult=1.0,
                        labor_productivity=1.0):
        """Sampling durasi dengan distribusi triangular + faktor risiko konstruksi"""
        base_duration = np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic,
            n_simulations
        )
        
        # ---- Faktor risiko cuaca buruk (discrete) ----
        for risk_name, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                probability = risk_params['probability']
                impact      = risk_params['impact']
                risk_occurs = np.random.random(n_simulations) < probability
                base_duration = np.where(
                    risk_occurs,
                    base_duration * (1 + impact),
                    base_duration
                )
            elif risk_params['type'] == 'continuous':
                mean = risk_params['mean']
                std  = risk_params['std']
                factor = np.random.normal(mean, std, n_simulations)
                base_duration = base_duration / np.clip(factor, 0.5, 2.0)
        
        # Terapkan multiplier global dari sidebar
        base_duration = base_duration * weather_risk_mult * material_risk_mult
        base_duration = base_duration / np.clip(
            np.random.normal(labor_productivity, 0.1, n_simulations), 0.5, 1.5
        )
        
        return base_duration


class MonteCarloConstructionSimulation:
    """Kelas untuk menjalankan simulasi Monte Carlo konstruksi gedung"""
    
    def __init__(self, phases_config, num_simulations=10000,
                 weather_risk_mult=1.0, material_risk_mult=1.0,
                 labor_productivity=1.0):
        self.phases_config       = phases_config
        self.num_simulations     = num_simulations
        self.weather_risk_mult   = weather_risk_mult
        self.material_risk_mult  = material_risk_mult
        self.labor_productivity  = labor_productivity
        self.stages              = {}
        self.simulation_results  = None
        self._initialize_phases()
        
    def _initialize_phases(self):
        for phase_name, config in self.phases_config.items():
            self.stages[phase_name] = ConstructionPhase(
                name=phase_name,
                base_params=config['base_params'],
                risk_factors=config.get('risk_factors', {}),
                dependencies=config.get('dependencies', [])
            )
    
    def run_simulation(self):
        """Menjalankan simulasi Monte Carlo"""
        results     = pd.DataFrame(index=range(self.num_simulations))
        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times   = pd.DataFrame(index=range(self.num_simulations))
        
        for phase_name, phase in self.stages.items():
            results[phase_name] = phase.sample_duration(
                self.num_simulations,
                weather_risk_mult  = self.weather_risk_mult,
                material_risk_mult = self.material_risk_mult,
                labor_productivity = self.labor_productivity
            )
        
        for phase_name in self.stages.keys():
            deps = self.stages[phase_name].dependencies
            if not deps:
                start_times[phase_name] = 0
            else:
                start_times[phase_name] = end_times[deps].max(axis=1)
            end_times[phase_name] = start_times[phase_name] + results[phase_name]
        
        results['Total_Duration'] = end_times.max(axis=1)
        
        for phase_name in self.stages.keys():
            results[f'{phase_name}_Start']  = start_times[phase_name]
            results[f'{phase_name}_Finish'] = end_times[phase_name]
        
        self.simulation_results = results
        return results
    
    def calculate_critical_path_probability(self):
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        total_duration = self.simulation_results['Total_Duration']
        critical_path_probs = {}
        
        for phase_name in self.stages.keys():
            stage_finish = self.simulation_results[f'{phase_name}_Finish']
            correlation  = self.simulation_results[phase_name].corr(total_duration)
            is_critical  = (stage_finish + 0.5) >= total_duration
            prob_critical = np.mean(is_critical)
            critical_path_probs[phase_name] = {
                'probability':   prob_critical,
                'correlation':   correlation,
                'avg_duration':  self.simulation_results[phase_name].mean()
            }
        
        return pd.DataFrame(critical_path_probs).T
    
    def analyze_risk_contribution(self):
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        total_var    = self.simulation_results['Total_Duration'].var()
        contributions = {}
        
        for phase_name in self.stages.keys():
            stage_var   = self.simulation_results[phase_name].var()
            stage_covar = self.simulation_results[phase_name].cov(
                self.simulation_results['Total_Duration']
            )
            contribution = (stage_covar / total_var) * 100
            contributions[phase_name] = {
                'variance':             stage_var,
                'contribution_percent': contribution,
                'std_dev':              np.sqrt(stage_var)
            }
        
        return pd.DataFrame(contributions).T

# ============================================================================
# 3. FUNGSI VISUALISASI
# ============================================================================
def create_distribution_plot(results, deadlines_ref):
    total_duration  = results['Total_Duration']
    mean_duration   = total_duration.mean()
    median_duration = np.median(total_duration)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=total_duration, nbinsx=60,
        name='Distribusi Durasi',
        marker_color='steelblue', opacity=0.75,
        histnorm='probability density'
    ))
    fig.add_vline(x=mean_duration,   line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_duration:.1f} bln")
    fig.add_vline(x=median_duration, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_duration:.1f} bln")
    
    ci_80 = np.percentile(total_duration, [10, 90])
    ci_95 = np.percentile(total_duration, [2.5, 97.5])
    fig.add_vrect(x0=ci_80[0], x1=ci_80[1], fillcolor="yellow",  opacity=0.15,
                  annotation_text="80% CI", line_width=0)
    fig.add_vrect(x0=ci_95[0], x1=ci_95[1], fillcolor="orange",  opacity=0.08,
                  annotation_text="95% CI", line_width=0)
    
    colors_dl = ["blue", "purple", "red"]
    for dl, col in zip(deadlines_ref, colors_dl):
        fig.add_vline(x=dl, line_dash="dot", line_color=col,
                      annotation_text=f"DL {dl} bln")
    
    fig.update_layout(
        title='Distribusi Durasi Total Pembangunan Gedung FITE',
        xaxis_title='Total Durasi (Bulan)',
        yaxis_title='Densitas Probabilitas',
        height=500
    )
    return fig, {'mean': mean_duration, 'median': median_duration,
                 'std': total_duration.std(),
                 'min': total_duration.min(), 'max': total_duration.max(),
                 'ci_80': ci_80, 'ci_95': ci_95}


def create_completion_probability_plot(results, deadlines_ref):
    total_duration = results['Total_Duration']
    lo = max(5, int(total_duration.min()) - 2)
    hi = int(total_duration.max()) + 3
    deadlines = np.arange(lo, hi + 1, 1)
    completion_probs = [np.mean(total_duration <= d) for d in deadlines]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=deadlines, y=completion_probs,
        mode='lines', name='Probabilitas Selesai',
        line=dict(color='darkblue', width=3),
        fill='tozeroy', fillcolor='rgba(173,216,230,0.3)'
    ))
    
    for lv, col, lbl in [(0.5, "red", "50%"), (0.8, "green", "80%"), (0.95, "blue", "95%")]:
        fig.add_hline(y=lv, line_dash="dash", line_color=col,
                      annotation_text=lbl, annotation_position="right")
    
    # Tandai ketiga deadline referensi
    colors_dl = ["blue", "purple", "red"]
    for dl, col in zip(deadlines_ref, colors_dl):
        prob = np.mean(total_duration <= dl)
        fig.add_trace(go.Scatter(
            x=[dl], y=[prob], mode='markers+text',
            marker=dict(size=14, color=col),
            text=[f'{dl} bln\n{prob:.1%}'],
            textposition="top center", showlegend=False
        ))
    
    fig.update_layout(
        title='Kurva Probabilitas Penyelesaian Proyek',
        xaxis_title='Deadline (Bulan)',
        yaxis_title='Probabilitas Selesai Tepat Waktu',
        yaxis_range=[-0.05, 1.05],
        height=500
    )
    return fig


def create_critical_path_plot(critical_analysis):
    critical_analysis = critical_analysis.sort_values('probability', ascending=True)
    colors = ['#DC2626' if p > 0.7 else '#F87171' for p in critical_analysis['probability']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[s.replace('_', ' ') for s in critical_analysis.index],
        x=critical_analysis['probability'],
        orientation='h', marker_color=colors,
        text=[f'{p:.1%}' for p in critical_analysis['probability']],
        textposition='auto'
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.7, line_dash="dot", line_color="orange")
    fig.update_layout(
        title='Probabilitas Tahapan Menjadi Critical Path',
        xaxis_title='Probabilitas Critical Path',
        xaxis_range=[0, 1.0], height=500
    )
    return fig


def create_stage_boxplot(results, stages):
    fig = go.Figure()
    for i, stage in enumerate(stages.keys()):
        fig.add_trace(go.Box(
            y=results[stage],
            name=stage.replace('_', '\n'),
            boxmean='sd',
            marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
            boxpoints='outliers'
        ))
    fig.update_layout(
        title='Distribusi Durasi per Tahapan Konstruksi',
        yaxis_title='Durasi (Bulan)', height=500, showlegend=False
    )
    return fig


def create_risk_contribution_plot(risk_contrib):
    risk_contrib = risk_contrib.sort_values('contribution_percent', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[n.replace('_', '\n') for n in risk_contrib.index],
        y=risk_contrib['contribution_percent'],
        marker_color=px.colors.qualitative.Set3,
        text=[f'{c:.1f}%' for c in risk_contrib['contribution_percent']],
        textposition='auto'
    ))
    fig.update_layout(
        title='Kontribusi Risiko per Tahapan terhadap Keterlambatan',
        yaxis_title='Kontribusi terhadap Variabilitas (%)', height=400
    )
    return fig


def create_resource_scenario_plot(results_base, results_resource, deadlines_ref):
    """Membandingkan distribusi skenario normal vs skenario tambah resource"""
    fig = go.Figure()
    for r, lbl, col in [
        (results_base,     'Normal',          'steelblue'),
        (results_resource, '+Resource',        'orange'),
    ]:
        fig.add_trace(go.Histogram(
            x=r['Total_Duration'], nbinsx=50,
            name=lbl, marker_color=col, opacity=0.6,
            histnorm='probability density'
        ))
    for dl, col in zip(deadlines_ref, ["blue", "purple", "red"]):
        fig.add_vline(x=dl, line_dash="dot", line_color=col,
                      annotation_text=f"{dl} bln")
    fig.update_layout(
        title='Perbandingan: Skenario Normal vs Penambahan Resource',
        xaxis_title='Total Durasi (Bulan)',
        yaxis_title='Densitas Probabilitas',
        barmode='overlay', height=450
    )
    return fig


def create_correlation_heatmap(results, stages):
    corr_matrix = results[list(stages.keys())].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[n.replace('_', '\n') for n in corr_matrix.columns],
        y=[n.replace('_', '\n') for n in corr_matrix.index],
        colorscale='RdBu', zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}', textfont={"size": 10}
    ))
    fig.update_layout(title='Matriks Korelasi Antar Tahapan', height=500)
    return fig


# ============================================================================
# 4. FUNGSI UTAMA STREAMLIT
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">🏗️ Simulasi Monte Carlo — Pembangunan Gedung FITE 5 Lantai</h1>',
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Gedung FITE 5 lantai dilengkapi dengan ruang kelas, lab komputer, lab elektro, lab mobile,
    lab VR/AR, lab game, ruang dosen, toilet, dan ruang serbaguna. Simulasi Monte Carlo digunakan
    untuk mengestimasi total durasi pembangunan secara probabilistik, mengidentifikasi tahapan kritis,
    dan menganalisis dampak berbagai faktor risiko terhadap jadwal.
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    st.sidebar.markdown('<h2>⚙️ Konfigurasi Simulasi</h2>', unsafe_allow_html=True)
    
    num_simulations = st.sidebar.slider(
        'Jumlah Iterasi Simulasi:',
        min_value=1000, max_value=50000, value=20000, step=1000
    )
    
    st.sidebar.markdown('---')
    st.sidebar.markdown('### 🌩️ Faktor Risiko Global')
    weather_prob = st.sidebar.slider(
        'Probabilitas Cuaca Buruk (per tahap):', 0.0, 0.8, 0.25, 0.05,
        help="Probabilitas gangguan cuaca buruk yang mempengaruhi pekerjaan"
    )
    material_delay_prob = st.sidebar.slider(
        'Probabilitas Keterlambatan Material:', 0.0, 0.8, 0.30, 0.05
    )
    labor_productivity = st.sidebar.slider(
        'Produktivitas Pekerja (1.0 = normal):', 0.6, 1.4, 1.0, 0.05
    )
    design_change_prob = st.sidebar.slider(
        'Probabilitas Perubahan Desain Lab:', 0.0, 0.6, 0.20, 0.05
    )
    
    # Deadline referensi (sesuai pertanyaan foto)
    st.sidebar.markdown('---')
    st.sidebar.markdown('### 📅 Skenario Deadline')
    dl1 = st.sidebar.number_input('Deadline Skenario 1 (bulan):', 5, 48, 16)
    dl2 = st.sidebar.number_input('Deadline Skenario 2 (bulan):', 5, 48, 20)
    dl3 = st.sidebar.number_input('Deadline Skenario 3 (bulan):', 5, 48, 24)
    deadlines_ref = [dl1, dl2, dl3]
    
    st.sidebar.markdown('---')
    st.sidebar.markdown('### 📋 Konfigurasi Durasi Tahapan')
    
    # ---- Konfigurasi default tahapan konstruksi gedung FITE ----
    default_config = {
        "Perencanaan_Desain": {
            "base_params": {"optimistic": 1.0, "most_likely": 1.5, "pessimistic": 2.5},
            "risk_factors": {
                "perubahan_desain_lab": {
                    "type": "discrete",
                    "probability": design_change_prob,
                    "impact": 0.40
                }
            },
            "dependencies": []
        },
        "Pondasi_Struktur": {
            "base_params": {"optimistic": 2.0, "most_likely": 3.0, "pessimistic": 5.0},
            "risk_factors": {
                "cuaca_buruk": {
                    "type": "discrete",
                    "probability": weather_prob,
                    "impact": 0.30
                },
                "keterlambatan_material": {
                    "type": "discrete",
                    "probability": material_delay_prob,
                    "impact": 0.25
                }
            },
            "dependencies": ["Perencanaan_Desain"]
        },
        "Struktur_Beton_5_Lantai": {
            "base_params": {"optimistic": 3.5, "most_likely": 5.0, "pessimistic": 8.0},
            "risk_factors": {
                "cuaca_buruk": {
                    "type": "discrete",
                    "probability": weather_prob,
                    "impact": 0.25
                },
                "material_teknis_khusus": {
                    "type": "discrete",
                    "probability": material_delay_prob,
                    "impact": 0.30
                },
                "produktivitas_pekerja": {
                    "type": "continuous",
                    "mean": labor_productivity,
                    "std": 0.12
                }
            },
            "dependencies": ["Pondasi_Struktur"]
        },
        "Dinding_Atap_Fasad": {
            "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.0},
            "risk_factors": {
                "cuaca_buruk": {
                    "type": "discrete",
                    "probability": weather_prob,
                    "impact": 0.20
                },
                "material_fasad": {
                    "type": "discrete",
                    "probability": material_delay_prob * 0.7,
                    "impact": 0.15
                }
            },
            "dependencies": ["Struktur_Beton_5_Lantai"]
        },
        "Instalasi_MEP": {
            "base_params": {"optimistic": 2.0, "most_likely": 3.0, "pessimistic": 5.5},
            "risk_factors": {
                "peralatan_teknis_khusus": {
                    "type": "discrete",
                    "probability": material_delay_prob,
                    "impact": 0.35
                },
                "produktivitas_teknisi": {
                    "type": "continuous",
                    "mean": labor_productivity,
                    "std": 0.15
                }
            },
            "dependencies": ["Dinding_Atap_Fasad"]
        },
        "Fit_Out_Laboratorium": {
            "base_params": {"optimistic": 2.5, "most_likely": 3.5, "pessimistic": 6.0},
            "risk_factors": {
                "perubahan_desain_lab": {
                    "type": "discrete",
                    "probability": design_change_prob,
                    "impact": 0.45
                },
                "material_lab_khusus": {
                    "type": "discrete",
                    "probability": material_delay_prob,
                    "impact": 0.40
                }
            },
            "dependencies": ["Instalasi_MEP"]
        },
        "Finishing_Uji_Serah_Terima": {
            "base_params": {"optimistic": 0.5, "most_likely": 1.0, "pessimistic": 2.0},
            "risk_factors": {
                "revisi_minor": {
                    "type": "discrete",
                    "probability": 0.35,
                    "impact": 0.50
                }
            },
            "dependencies": ["Fit_Out_Laboratorium"]
        }
    }
    
    # Input parameter per tahapan di sidebar
    for phase_name, config in default_config.items():
        with st.sidebar.expander(f"🏗️ {phase_name.replace('_', ' ')}", expanded=False):
            opt = st.number_input("Optimistic (bln)", 0.5, 20.0,
                                  float(config['base_params']['optimistic']), 0.5,
                                  key=f"opt_{phase_name}")
            ml  = st.number_input("Most Likely (bln)", 0.5, 20.0,
                                  float(config['base_params']['most_likely']),  0.5,
                                  key=f"ml_{phase_name}")
            pes = st.number_input("Pessimistic (bln)", 0.5, 30.0,
                                  float(config['base_params']['pessimistic']), 0.5,
                                  key=f"pes_{phase_name}")
            default_config[phase_name]['base_params'] = {
                'optimistic': opt, 'most_likely': ml, 'pessimistic': pes
            }
    
    run_simulation = st.sidebar.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)
    
    # Resource scenario (untuk pertanyaan 5)
    st.sidebar.markdown('---')
    st.sidebar.markdown('### ➕ Skenario Penambahan Resource')
    resource_productivity_boost = st.sidebar.slider(
        'Peningkatan Produktivitas dg Tambahan Resource:', 1.0, 1.8, 1.25, 0.05,
        help="Misal 1.25 = produktivitas naik 25% dengan penambahan pekerja/alat berat/insinyur"
    )
    resource_material_reduction = st.sidebar.slider(
        'Pengurangan Risiko Material (faktor):', 0.3, 1.0, 0.6, 0.05,
        help="Misal 0.6 = probabilitas keterlambatan material berkurang 40%"
    )
    
    # ---- Session State ----
    for key in ['sim_results', 'simulator', 'sim_results_resource', 'simulator_resource']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    if run_simulation:
        with st.spinner('Menjalankan simulasi Monte Carlo... Harap tunggu...'):
            # Simulasi Normal
            sim = MonteCarloConstructionSimulation(
                phases_config       = default_config,
                num_simulations     = num_simulations,
                weather_risk_mult   = 1.0,
                material_risk_mult  = 1.0,
                labor_productivity  = labor_productivity
            )
            results = sim.run_simulation()
            st.session_state.sim_results = results
            st.session_state.simulator   = sim
            
            # Simulasi Skenario + Resource
            # Buat config dengan material risk dikurangi
            import copy
            config_resource = copy.deepcopy(default_config)
            for pn, cfg in config_resource.items():
                for rf_name, rf_val in cfg.get('risk_factors', {}).items():
                    if 'material' in rf_name or 'keterlambatan' in rf_name:
                        rf_val['probability'] = rf_val['probability'] * resource_material_reduction
            
            sim_res = MonteCarloConstructionSimulation(
                phases_config       = config_resource,
                num_simulations     = num_simulations,
                weather_risk_mult   = 1.0,
                material_risk_mult  = 1.0,
                labor_productivity  = resource_productivity_boost
            )
            results_resource = sim_res.run_simulation()
            st.session_state.sim_results_resource = results_resource
            st.session_state.simulator_resource   = sim_res
            
        st.success(f'✅ Simulasi selesai! {num_simulations:,} iterasi berhasil dijalankan.')
    
    # ---- Tampilkan Hasil ----
    if st.session_state.sim_results is not None:
        results          = st.session_state.sim_results
        simulator        = st.session_state.simulator
        results_resource = st.session_state.sim_results_resource
        
        total_duration  = results['Total_Duration']
        mean_duration   = total_duration.mean()
        median_duration = np.median(total_duration)
        ci_80 = np.percentile(total_duration, [10, 90])
        ci_95 = np.percentile(total_duration, [2.5, 97.5])
        
        # ==== JAWABAN PERTANYAAN 1: Total waktu ====
        st.markdown('<h2 class="sub-header">📌 P1 — Berapa Total Waktu Penyelesaian Proyek?</h2>',
                    unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            (f"{mean_duration:.1f} bln",              "Rata-rata Durasi"),
            (f"{median_duration:.1f} bln",             "Median Durasi"),
            (f"{ci_80[0]:.1f}–{ci_80[1]:.1f} bln",   "80% Confidence Interval"),
            (f"{ci_95[0]:.1f}–{ci_95[1]:.1f} bln",   "95% Confidence Interval"),
        ]
        for col, (val, lbl) in zip([col1, col2, col3, col4], metrics):
            col.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{lbl}</p></div>',
                         unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        fig_dist, stats = create_distribution_plot(results, deadlines_ref)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        with st.expander("📋 Detail Statistik Distribusi"):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"- Rata-rata : **{stats['mean']:.2f} bulan**")
                st.write(f"- Median    : **{stats['median']:.2f} bulan**")
                st.write(f"- Std Dev   : **{stats['std']:.2f} bulan**")
                st.write(f"- Min       : **{stats['min']:.2f} bulan**")
                st.write(f"- Max       : **{stats['max']:.2f} bulan**")
            with c2:
                st.write(f"- 80% CI : [{stats['ci_80'][0]:.2f}, {stats['ci_80'][1]:.2f}] bln")
                st.write(f"- 95% CI : [{stats['ci_95'][0]:.2f}, {stats['ci_95'][1]:.2f}] bln")
        
        st.markdown("---")
        
        # ==== JAWABAN PERTANYAAN 2: Risiko keterlambatan ====
        st.markdown('<h2 class="sub-header">📌 P2 — Risiko Keterlambatan akibat Faktor Ketidakpastian</h2>',
                    unsafe_allow_html=True)
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            risk_contrib = simulator.analyze_risk_contribution()
            fig_risk = create_risk_contribution_plot(risk_contrib)
            st.plotly_chart(fig_risk, use_container_width=True)
        with col_r2:
            fig_corr = create_correlation_heatmap(results, simulator.stages)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        prob_delay_base = np.mean(total_duration > mean_duration)
        prob_exceed_p95 = np.mean(total_duration > ci_95[1])
        
        st.markdown(f"""
        <div class="warning-box">
        <b>Ringkasan Risiko Keterlambatan:</b><br>
        • Probabilitas melebihi rata-rata estimasi: <b>{prob_delay_base:.1%}</b><br>
        • Probabilitas melebihi batas 95% CI ({ci_95[1]:.1f} bln): <b>{prob_exceed_p95:.1%}</b><br>
        • Variabilitas terbesar: <b>{risk_contrib['std_dev'].idxmax().replace('_',' ')}</b>
          (std dev = {risk_contrib['std_dev'].max():.2f} bln)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ==== JAWABAN PERTANYAAN 3: Critical Path ====
        st.markdown('<h2 class="sub-header">📌 P3 — Tahapan yang Paling Kritis (Critical Path)</h2>',
                    unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns(2)
        critical_analysis = simulator.calculate_critical_path_probability()
        with col_c1:
            fig_cp = create_critical_path_plot(critical_analysis)
            st.plotly_chart(fig_cp, use_container_width=True)
        with col_c2:
            fig_bp = create_stage_boxplot(results, simulator.stages)
            st.plotly_chart(fig_bp, use_container_width=True)
        
        most_critical = critical_analysis['probability'].idxmax()
        st.markdown(f"""
        <div class="info-box">
        <b>Tahapan Paling Kritis:</b> <span class="risk-high">{most_critical.replace('_',' ')}</span>
        dengan probabilitas menjadi critical path = 
        <b>{critical_analysis.loc[most_critical,'probability']:.1%}</b><br>
        Tahapan ini paling menentukan durasi total proyek — keterlambatan di sini 
        langsung berdampak pada jadwal selesai gedung FITE.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Tabel Lengkap Analisis Critical Path"):
            st.dataframe(
                critical_analysis.sort_values('probability', ascending=False)
                    .style.format({'probability': '{:.1%}', 'correlation': '{:.3f}',
                                   'avg_duration': '{:.2f}'}),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # ==== JAWABAN PERTANYAAN 4: Probabilitas vs Deadline ====
        st.markdown('<h2 class="sub-header">📌 P4 — Probabilitas Penyelesaian Sesuai Skenario Deadline</h2>',
                    unsafe_allow_html=True)
        
        fig_prob = create_completion_probability_plot(results, deadlines_ref)
        st.plotly_chart(fig_prob, use_container_width=True)
        
        col_d = st.columns(len(deadlines_ref))
        for i, dl in enumerate(deadlines_ref):
            prob_on_time = np.mean(total_duration <= dl)
            prob_late    = 1 - prob_on_time
            col_d[i].metric(
                label=f"Deadline {dl} Bulan",
                value=f"{prob_on_time:.1%}",
                delta=f"{prob_late:.1%} risiko terlambat",
                delta_color="inverse"
            )
        
        # Deadline custom
        st.markdown("**Cek Deadline Kustom:**")
        target_deadline = st.number_input(
            "Masukkan deadline target (bulan):",
            min_value=1.0, max_value=60.0, value=float(deadlines_ref[1]), step=0.5
        )
        prob_target = np.mean(total_duration <= target_deadline)
        st.metric(
            label=f"Probabilitas selesai dalam {target_deadline:.1f} bulan",
            value=f"{prob_target:.1%}",
            delta=f"Risiko terlambat: {1-prob_target:.1%}",
            delta_color="inverse"
        )
        
        st.markdown("---")
        
        # ==== JAWABAN PERTANYAAN 5: Penambahan Resource ====
        st.markdown('<h2 class="sub-header">📌 P5 — Dampak Penambahan Resource terhadap Percepatan</h2>',
                    unsafe_allow_html=True)
        
        mean_resource   = results_resource['Total_Duration'].mean()
        median_resource = np.median(results_resource['Total_Duration'])
        ci80_resource   = np.percentile(results_resource['Total_Duration'], [10, 90])
        
        fig_compare = create_resource_scenario_plot(results, results_resource, deadlines_ref)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        percepatan = mean_duration - mean_resource
        pct_percepatan = (percepatan / mean_duration) * 100
        
        cols5 = st.columns(3)
        scenarios = [
            ("Normal",        f"{mean_duration:.1f} bln",   f"CI80: {ci_80[0]:.1f}–{ci_80[1]:.1f}"),
            ("+Resource",     f"{mean_resource:.1f} bln",   f"CI80: {ci80_resource[0]:.1f}–{ci80_resource[1]:.1f}"),
            ("Percepatan",    f"{percepatan:.1f} bln",      f"({pct_percepatan:.1f}% lebih cepat)"),
        ]
        for col, (lbl, val, sub) in zip(cols5, scenarios):
            col.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{lbl}<br><small>{sub}</small></p></div>',
                         unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        prob_cols = st.columns(len(deadlines_ref))
        for i, dl in enumerate(deadlines_ref):
            p_norm = np.mean(total_duration <= dl)
            p_res  = np.mean(results_resource['Total_Duration'] <= dl)
            prob_cols[i].metric(
                label=f"P(selesai ≤ {dl} bln)",
                value=f"+Resource: {p_res:.1%}",
                delta=f"Normal: {p_norm:.1%} → naik {p_res-p_norm:.1%}",
                delta_color="normal"
            )
        
        st.markdown(f"""
        <div class="info-box">
        <b>Kesimpulan Penambahan Resource:</b><br>
        Dengan menambah pekerja khusus, alat berat, dan insinyur (produktivitas ×{resource_productivity_boost}
        dan risiko material ×{resource_material_reduction}), estimasi rata-rata durasi berkurang dari
        <b>{mean_duration:.1f} bulan</b> menjadi <b>{mean_resource:.1f} bulan</b>
        (<b>percepatan {percepatan:.1f} bulan / {pct_percepatan:.1f}%</b>).
        </div>
        """, unsafe_allow_html=True)
        
        # ---- Ringkasan 5 Pertanyaan ----
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Ringkasan Jawaban 5 Pertanyaan</h2>', unsafe_allow_html=True)
        summary_data = {
            "Pertanyaan": [
                "P1: Total waktu penyelesaian",
                "P2: Risiko keterlambatan",
                "P3: Tahapan paling kritis",
                f"P4a: P(selesai ≤ {dl1} bln)",
                f"P4b: P(selesai ≤ {dl2} bln)",
                f"P4c: P(selesai ≤ {dl3} bln)",
                "P5: Percepatan dg tambahan resource",
            ],
            "Hasil": [
                f"Rata-rata {mean_duration:.1f} bln | 80% CI: {ci_80[0]:.1f}–{ci_80[1]:.1f} bln",
                f"Variabilitas terbesar: {risk_contrib['std_dev'].idxmax().replace('_',' ')} | Std dev total: {total_duration.std():.2f} bln",
                f"{most_critical.replace('_',' ')} (P kritis = {critical_analysis.loc[most_critical,'probability']:.1%})",
                f"{np.mean(total_duration <= dl1):.1%}",
                f"{np.mean(total_duration <= dl2):.1%}",
                f"{np.mean(total_duration <= dl3):.1%}",
                f"Percepatan {percepatan:.1f} bln ({pct_percepatan:.1f}%): {mean_duration:.1f} → {mean_resource:.1f} bln",
            ]
        }
        st.table(pd.DataFrame(summary_data))
    
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem; background-color:#f8f9fa; border-radius:10px;">
            <h3>🏗️ Siap untuk memulai simulasi?</h3>
            <p>Atur parameter di sidebar, lalu klik <b>"Jalankan Simulasi"</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">📋 Preview Konfigurasi Tahapan</h2>', unsafe_allow_html=True)
        for phase_name, config in default_config.items():
            b = config['base_params']
            st.markdown(f"""
            <div class="stage-card">
            <b>{phase_name.replace('_', ' ')}</b> |
            Optimistic: {b['optimistic']} bln |
            Most Likely: {b['most_likely']} bln |
            Pessimistic: {b['pessimistic']} bln
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#666; font-size:0.85rem;">
    <b>Simulasi Monte Carlo — Pembangunan Gedung FITE 5 Lantai</b><br>
    ⚠️ Hasil simulasi merupakan estimasi probabilistik, bukan prediksi pasti.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()