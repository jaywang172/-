import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, get_window, find_peaks
import matplotlib.pyplot as plt


# --- 1. 核心分析函數 (增加了wavenumber_range參數) ---
def analyze_spectrum(wavenumber, reflectance, sample_name="Sample", wavenumber_range=None):
    """
    對給定的反射光譜進行傅立葉變換分析，提取光程差(OPD)。
    """
    print(f"\n--- Analyzing {sample_name} ---")

    # 【新功能】根據提供的範圍過濾數據
    if wavenumber_range:
        mask = (wavenumber >= wavenumber_range[0]) & (wavenumber <= wavenumber_range[1])
        wavenumber = wavenumber[mask]
        reflectance = reflectance[mask]
        print(f"Analysis restricted to wavenumber range: {wavenumber_range[0]}-{wavenumber_range[1]} cm⁻¹")

    # 步驟 1: 數據預處理 - 均勻重採樣
    min_wn, max_wn = np.min(wavenumber), np.max(wavenumber)
    num_points = len(wavenumber)
    uniform_wn = np.linspace(min_wn, max_wn, num_points)
    delta_nu = uniform_wn[1] - uniform_wn[0]
    interp_func = interp1d(wavenumber, reflectance, kind='linear', fill_value="extrapolate")
    uniform_reflectance = interp_func(uniform_wn)

    # 步驟 2: 基線校正
    window_length = int(num_points / 5)  # 調整窗口以適應可能更短的數據範圍
    if window_length % 2 == 0: window_length += 1
    baseline = savgol_filter(uniform_reflectance, window_length, 3)
    interference_signal = uniform_reflectance - baseline

    # 步驟 3: 應用窗函數
    window = get_window('hann', num_points)
    windowed_signal = interference_signal * window

    # 步驟 4: 快速傅立葉變換
    n_fft = num_points * 8
    fft_result = np.fft.fft(windowed_signal, n=n_fft)
    opd_axis = np.fft.fftfreq(n_fft, d=delta_nu)
    power_spectrum = np.abs(fft_result) ** 2

    positive_mask = opd_axis > 0
    opd_axis = opd_axis[positive_mask]
    power_spectrum = power_spectrum[positive_mask]

    # 步驟 5: 提取主峰位置
    search_start_idx = np.where(opd_axis > 1e-4)[0][0]
    peak_idx = search_start_idx + np.argmax(power_spectrum[search_start_idx:])
    opd_peak = opd_axis[peak_idx]

    print(f"Detected Optical Path Difference (OPD): {opd_peak * 1e4:.4f} μm")

    # --- 可視化 ---
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(uniform_wn, uniform_reflectance, label='(Filtered) Interpolated Reflectance')
    plt.plot(uniform_wn, baseline, label='Fitted Baseline', linestyle='--')
    plt.plot(uniform_wn, interference_signal, label='Interference Signal', alpha=0.7)
    plt.title(f'Reflectance Spectrum and Signal Processing for {sample_name}')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Reflectance (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(opd_axis * 1e4, power_spectrum)
    plt.axvline(opd_peak * 1e4, color='r', linestyle='--', label=f'Peak OPD = {opd_peak * 1e4:.2f} μm')
    plt.title(f'Optical Path Difference (OPD) Spectrum for {sample_name}')
    plt.xlabel('Optical Path Difference (μm)')
    plt.ylabel('Power (arbitrary units)')
    plt.xlim(0, opd_peak * 1e4 * 3)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return opd_peak, opd_axis, power_spectrum


# --- 2. 主執行流程 ---
if __name__ == "__main__":
    file_paths = {
        'sic_10deg': '附件1.xlsx', 'sic_15deg': '附件2.xlsx',
        'si_10deg': '附件3.xlsx', 'si_15deg': '附件4.xlsx',
    }

    data = {}
    for name, path in file_paths.items():
        try:
            df = pd.read_excel(path, skiprows=1, header=None, names=['wavenumber', 'reflectance'])
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            data[name] = {'wavenumber': df['wavenumber'].values, 'reflectance': df['reflectance'].values}
        except FileNotFoundError:
            print(f"Error: File not found at '{path}'. Using dummy data.")
            data[name] = {'wavenumber': np.linspace(400, 1000, 1000),
                          'reflectance': 30 + 5 * np.sin(np.linspace(400, 1000, 1000) * 0.1)}

    # --- 分析碳化硅(SiC)樣品，【修改點】增加了 wavenumber_range ---
    # 我們只分析 1200 cm⁻¹ 以上的數據，以避開SiC的強吸收區
    valid_range_sic = (1200, 4000)
    opd1, _, _ = analyze_spectrum(data['sic_10deg']['wavenumber'], data['sic_10deg']['reflectance'], "SiC Sample (10°)",
                                  wavenumber_range=valid_range_sic)
    opd2, _, _ = analyze_spectrum(data['sic_15deg']['wavenumber'], data['sic_15deg']['reflectance'], "SiC Sample (15°)",
                                  wavenumber_range=valid_range_sic)

    theta1_rad, theta2_rad = np.deg2rad(10), np.deg2rad(15)
    sin2_th1, sin2_th2 = np.sin(theta1_rad) ** 2, np.sin(theta2_rad) ** 2

    A = np.array([[1, -sin2_th1], [1, -sin2_th2]])
    b = np.array([opd1 ** 2, opd2 ** 2])

    try:
        solution = np.linalg.solve(A, b)
        X, Y = solution[0], solution[1]
        if Y > 0 and X / Y > 0:
            d_um = (np.sqrt(Y) / 2) * 1e4
            n1_eff = np.sqrt(X / Y)
            print("\n--- Final Results for SiC Sample (Refined) ---")
            print(f"Calculated Thickness (d): {d_um:.4f} μm")
            print(f"Calculated Effective Refractive Index (n1): {n1_eff:.4f}")
        else:
            print("\nError: Calculation resulted in a non-physical solution after data filtering.")
    except np.linalg.LinAlgError:
        print("\nError: Could not solve the linear system with filtered data.")

    # --- 分析硅(Si)樣品 (附件3和4) - 診斷多光束干涉 ---
    opd_si1, opd_axis_si, power_spectrum_si = analyze_spectrum(data['si_10deg']['wavenumber'],
                                                               data['si_10deg']['reflectance'], "Si Sample (10°)")

    print("\n--- Multi-beam Interference Analysis for Si Sample ---")

    # 【修改點】在主腳本中重新安全地找到主峰的索引
    search_start_idx = np.where(opd_axis_si > 1e-4)[0][0]
    main_peak_idx = search_start_idx + np.argmax(power_spectrum_si[search_start_idx:])

    # 使用scipy.signal.find_peaks來識別所有顯著的峰
    peaks, _ = find_peaks(power_spectrum_si, height=np.max(power_spectrum_si) * 0.01, distance=main_peak_idx / 2)

    if len(peaks) > 1:
        print("Multiple significant peaks found in OPD spectrum, suggesting multi-beam interference.")
        peak_opds_um = opd_axis_si[peaks] * 1e4

        # 確保峰值按位置排序
        sorted_indices = np.argsort(peak_opds_um)
        peak_opds_um = peak_opds_um[sorted_indices]

        peak_ratios = peak_opds_um / peak_opds_um[0]

        print("Peak Positions (μm):", [f"{p:.2f}" for p in peak_opds_um])
        print("Peak Position Ratios to Fundamental:", [f"{r:.2f}" for r in peak_ratios])

        is_harmonic = any(np.isclose(ratio, round(ratio), atol=0.1) and round(ratio) > 1 for ratio in peak_ratios)

        if is_harmonic:
            print(
                "\nConclusion: The presence of peaks at integer multiples of the fundamental OPD is a strong indicator of Fabry-Pérot (multi-beam) interference.")
        else:
            print("\nConclusion: Multiple peaks detected, but they do not appear to be simple harmonics.")
    else:
        print(
            "Only one dominant peak found. Multi-beam interference is not strongly evident from this analysis alone, but the high reflectance values strongly suggest its presence.")