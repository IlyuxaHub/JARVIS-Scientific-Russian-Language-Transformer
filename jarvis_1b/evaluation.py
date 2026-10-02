"""Fixed text suites. Generated answers are observations, never automatic correctness labels."""
TOKENIZER_EVAL_SAMPLES = {
    'english_science': (
        'For a closed system, the first law of thermodynamics states that '
        'the change in internal energy equals heat supplied minus work done by the system.'),
    'russian_science': (
        'Для замкнутой системы первый закон термодинамики связывает изменение '
        'внутренней энергии с теплотой и совершённой работой.'),
    'mixed_bilingual': (
        'The eigenvalue λ определяет масштабирование eigenvector v: A v = λ v. '
        'Единицы измерения энергии: joule (Дж).'),
    'mathematics_latex': (
        r'Let $A=A^\top\in\mathbb{R}^{n\times n}$. Then '
        r'$A=Q\Lambda Q^\top$, and $\int_0^\infty e^{-x^2}\,dx=\sqrt{\pi}/2$.'),
    'code': (
        'def conjugate_gradient(A, b, tol=1e-8):\n'
        '    """Solve Ax=b for symmetric positive-definite A."""\n'
        '    x = np.zeros_like(b)\n    r = b - A @ x\n    return x\n'),
    'numbers_units': 'c = 299792458 m/s; ℏ = 1.054571817×10⁻³⁴ J·s; T = −273.15 °C; 6.02214076e23 mol⁻¹.',
    'unicode_whitespace': '  α β γ ∇ ∂ ∑ ∫ √ ≤ ≥ ℝ ℂ\r\n\tindent  two spaces\n\nконец\u00a0\u202f🧪',
}


QUALITATIVE_PROMPTS_VERSION = 1
