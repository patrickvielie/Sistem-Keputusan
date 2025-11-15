from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

NUM_CRITERIA = 5

# ======================= Util umum =======================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def normalize_weights_to_1(weights):
    s = sum(weights)
    if s == 0:
        raise ValueError("Jumlah bobot tidak boleh 0.")
    return [w / s for w in weights]

def ranking_from_scores(names, scores, desc=True):
    return sorted(
        [{"name": names[i], "score": scores[i]} for i in range(len(names))],
        key=lambda x: x["score"],
        reverse=desc
    )

# ========================= SAW ===========================
def compute_saw(names, matrix, weights):
    m = NUM_CRITERIA
    # 1) Max per kriteria
    max_vals = []
    for j in range(m):
        col = [row[j] for row in matrix]
        max_vals.append(max(col) if col else 0.0)

    # 2) Normalisasi benefit (x / max)
    norm = []
    for row in matrix:
        nr = [(row[j] / max_vals[j]) if max_vals[j] != 0 else 0.0 for j in range(m)]
        norm.append(nr)

    # 3) Bobot × normalisasi & skor
    weighted = []
    scores = []
    for nr in norm:
        wr = [nr[j] * weights[j] for j in range(m)]
        vi = sum(wr)
        weighted.append(wr)
        scores.append(vi)

    return {
        "max_vals": max_vals,
        "norm": norm,
        "weighted": weighted,
        "scores": scores,
        "ranking": ranking_from_scores(names, scores),
    }

# ========================== WP ===========================
def compute_wp(names, matrix, weights, cost_flags):
    if len(weights) != NUM_CRITERIA:
        raise ValueError(f"WP: Bobot harus {NUM_CRITERIA} angka.")
    if len(cost_flags) != NUM_CRITERIA:
        raise ValueError(f"WP: Cost flags harus {NUM_CRITERIA} angka (0/1).")

    m = NUM_CRITERIA
    exps = [weights[j] if int(cost_flags[j]) == 0 else -weights[j] for j in range(m)]

    S = []
    for row in matrix:
        s_val = 1.0
        for j in range(m):
            x = max(safe_float(row[j], 0.0), 1e-12)  # hindari 0^(-w)
            s_val *= x ** exps[j]
        S.append(s_val)

    total = sum(S)
    V = [s / total if total != 0 else 0.0 for s in S]

    return {
        "exponents": exps,
        "S": S,
        "V": V,
        "ranking": ranking_from_scores(names, V),
    }

# ========================== AHP (utilities) ==========================
def ahp_col_norm(matrix):
    """Normalisasi kolom untuk matriks pairwise (kembalikan matrix_norm, col_sums)."""
    arr = np.array(matrix, dtype=float)
    col_sum = arr.sum(axis=0)
    col_sum[col_sum == 0] = 1.0
    norm = arr / col_sum
    return norm.tolist(), col_sum.tolist()

def ahp_details(matrix):
    """
    Kembalikan paket lengkap:
      - A (as list), norm (kolom), weights (eigen approx via avg row),
      - lambda_max, CI, CR, col_sums
    """
    A = np.array(matrix, dtype=float)
    norm, col_sums = ahp_col_norm(A)
    w = np.array(norm).mean(axis=1)
    if w.sum() != 0:
        w = w / w.sum()
    lam = np.mean((A.dot(w)) / (w + 1e-12))
    n = A.shape[0]
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI_dict = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0.0
    return {
        "A": A.tolist(),
        "norm": norm,
        "col_sums": col_sums,
        "weights": w.astype(float).tolist(),
        "lambda_max": float(lam),
        "CI": float(CI),
        "CR": float(CR),
    }

def pairwise_from_weights(ws):
    """Buat matriks pairwise konsisten dari bobot ws (ws>0)."""
    n = len(ws)
    A = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = (ws[i] / ws[j]) if ws[j] != 0 else 1.0
            A[i, j] = val
            A[j, i] = 1.0 / val
    return A

def eigen_approx_from_matrix_power(A):
    """
    Hitung aproksimasi eigenvector berdasarkan metode yang dipakai di contoh:
    - M2 = A @ A -> hitung jumlah baris & normalisasi (row_sums1 / total1)
    - M4 = M2 @ M2 -> same
    Kembalikan semua nilai berguna.
    """
    A = np.array(A, dtype=float)
    M2 = A @ A
    row_sums1 = M2.sum(axis=1)
    total1 = row_sums1.sum() if row_sums1.sum() != 0 else 1.0
    eig_m2 = row_sums1 / total1

    M4 = M2 @ M2
    row_sums2 = M4.sum(axis=1)
    total2 = row_sums2.sum() if row_sums2.sum() != 0 else 1.0
    eig_m4 = row_sums2 / total2

    diff = eig_m2 - eig_m4
    return {
        "M2": M2.tolist(),
        "row_sums_m2": row_sums1.tolist(),
        "total_m2": float(total1),
        "eig_m2": eig_m2.tolist(),
        "M4": M4.tolist(),
        "row_sums_m4": row_sums2.tolist(),
        "total_m4": float(total2),
        "eig_m4": eig_m4.tolist(),
        "diff": diff.tolist(),
    }

# ======================== ROUTES =========================
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------- SAW -----------------------------
@app.route("/saw", methods=["GET", "POST"])
def saw():
    context = {
        "matrix_input": None,
        "norm": None,
        "weighted": None,
        "ranking": None,
        "error": None,
        "weights": None,
        "scores": None,
        "max_vals": None,
    }

    if request.method == "POST":
        try:
            kandidat = int(request.form.get("kandidat", "0"))
            if kandidat < 1:
                raise ValueError("Jumlah film harus minimal 1.")

            bobot_raw = request.form.get("bobot", "").strip()
            weights = [safe_float(x) for x in bobot_raw.split(",") if x.strip() != ""]
            if len(weights) != NUM_CRITERIA:
                raise ValueError(f"Masukkan {NUM_CRITERIA} bobot dipisah koma (contoh: 0.3,0.2,0.2,0.2,0.1).")
            weights = normalize_weights_to_1(weights)

            names, matrix = [], []
            for i in range(1, kandidat + 1):
                name = request.form.get(f"nama_{i}", "").strip() or f"Film {i}"
                names.append(name)
                row = []
                for j in range(1, NUM_CRITERIA + 1):
                    val_raw = request.form.get(f"c_{i}_{j}", "").strip()
                    row.append(safe_float(val_raw, 0.0))
                matrix.append(row)

            res = compute_saw(names, matrix, weights)
            context.update({
                "weights": weights,
                "matrix_input": [{"name": names[i], "vals": matrix[i]} for i in range(len(names))],
                "norm": [{"name": names[i], "vals": res["norm"][i]} for i in range(len(names))],
                "weighted": [{"name": names[i], "vals": res["weighted"][i], "vi": res["scores"][i]} for i in range(len(names))],
                "scores": res["scores"],
                "ranking": res["ranking"],
                "max_vals": res["max_vals"],
            })
        except Exception as e:
            context["error"] = str(e)

    return render_template("saw.html", **context)

# ------------------------ WP -----------------------------
@app.route("/wp", methods=["GET", "POST"])
def wp():
    context = {
        "error": None,
        "default_weights": None,
        "names": None,
        "matrix_input": None,
        "weights": None,
        "cost_flags": None,
        "exponents": None,
        "S": None,
        "V": None,
        "ranking": None,
    }

    if request.method == "POST":
        try:
            kandidat = int(request.form.get("kandidat", "0"))
            if kandidat < 1:
                raise ValueError("Jumlah kandidat minimal 1.")

            wraw = [s for s in request.form.get("bobot", "").split(",") if s.strip() != ""]
            if len(wraw) != NUM_CRITERIA:
                raise ValueError(f"Masukkan {NUM_CRITERIA} bobot dipisah koma.")
            weights = normalize_weights_to_1([safe_float(x) for x in wraw])

            cfraw = [s for s in request.form.get("cost_flags", "0,0,0,0,0").split(",") if s.strip() != ""]
            if len(cfraw) != NUM_CRITERIA:
                raise ValueError(f"Masukkan {NUM_CRITERIA} cost flags (0=benefit,1=cost).")
            cost_flags = [int(float(x)) for x in cfraw]

            names, matrix = [], []
            for i in range(1, kandidat + 1):
                names.append(request.form.get(f"nama_{i}", f"Kandidat {i}"))
                row = [safe_float(request.form.get(f"c_{i}_{j}", "0")) for j in range(1, NUM_CRITERIA + 1)]
                matrix.append(row)

            wp_res = compute_wp(names, matrix, weights, cost_flags)
            context.update({
                "names": names,
                "matrix_input": [{"name": names[i], "vals": matrix[i]} for i in range(len(names))],
                "weights": weights,
                "cost_flags": cost_flags,
                "exponents": wp_res["exponents"],
                "S": wp_res["S"],
                "V": wp_res["V"],
                "ranking": wp_res["ranking"],
                "default_weights": ",".join(str(round(w, 6)) for w in weights),
            })
        except Exception as e:
            context["error"] = str(e)

    return render_template("wp.html", **context)

# --------------------- AHP (full process ) ---------
@app.route("/ahp", methods=["GET", "POST"])
def ahp():
    # nama kriteria dan alternatif (10 alternatif yang saya tentukan)
    criteria_names = ["Genre", "Rating IMDb", "Rating Usia", "Durasi", "Tahun Rilis"]
    alternatives = [
        "Avengers: Endgame", "Inception", "The Dark Knight", "Interstellar", "Parasite",
        "Toy Story 4", "Joker", "Spider-Man: Across the Spider-Verse",
        "The Social Network", "Oppenheimer"
    ]

    # ✅ Default matriks pairwise sesuai permintaanmu
    suggested_pairwise = np.array([
        [1,   3,   5,   3,   7],
        [1/3, 1,   3,   2,   5],
        [1/5, 1/3, 1, 0.5,   3],
        [1/3, 0.5, 2,   1,   3],
        [1/7, 1/5, 1/3, 1/3, 1]
    ], dtype=float)

    context = {
        "error": None,
        "criteria_names": criteria_names,
        "alternatives": alternatives,
        "default_pairwise": suggested_pairwise.tolist(),

        # outputs (default None)
        "crit_matrix": None,
        "crit_power": None,
        "crit_det": None,
        "criteria_weights": None,
        "criteria_lambda_max": None,
        "criteria_CI": None,
        "criteria_CR": None,
        "alt_blocks": None,
        "alt_priority_matrix": None,
        "global_scores": None,
        "ranking": None,
    }

    

    if request.method == "POST":
        try:
            n = len(alternatives)  # kita pakai 10 alternatif otomatis
            # =========== baca matriks pairwise kriteria dari form ===========
            Acrit = np.zeros((NUM_CRITERIA, NUM_CRITERIA), dtype=float)
            for i in range(NUM_CRITERIA):
                for j in range(NUM_CRITERIA):
                    Acrit[i, j] = safe_float(request.form.get(f"c_{i}_{j}", ""), 0.0)
            # Jika user memasukkan 0/blank di triangular, perbaiki jadi reciprocal/1
            # (but template memberikan semua sel — still guard)
            for i in range(NUM_CRITERIA):
                for j in range(NUM_CRITERIA):
                    if i == j and (Acrit[i, j] == 0 or np.isnan(Acrit[i,j])):
                        Acrit[i, j] = 1.0
            # ensure reciprocity roughly: if A[i,j]==0 and A[j,i]!=0, set A[i,j]=1/A[j,i]
            for i in range(NUM_CRITERIA):
                for j in range(NUM_CRITERIA):
                    if Acrit[i, j] == 0 and Acrit[j, i] != 0:
                        Acrit[i, j] = 1.0 / Acrit[j, i]
                    if Acrit[i, j] == 0 and i == j:
                        Acrit[i, j] = 1.0

            # =========== nilai bobot kriteria (eigen aprox via matrix power) ===========
            crit_power = eigen_approx_from_matrix_power(Acrit)
            crit_eig_m4 = np.array(crit_power["eig_m4"], dtype=float)
            if crit_eig_m4.sum() != 0:
                crit_weights_final = crit_eig_m4 / crit_eig_m4.sum()
            else:
                crit_weights_final = np.ones(NUM_CRITERIA) / NUM_CRITERIA

            # use ahp_details for lambda_max/CI/CR (the standard column-normalization approx)
            crit_det = ahp_details(Acrit)

            # =========== buat raw skor alternatif sesuai pemetaan C1..C5 (saya tentukan) ===========
            # mapping functions mengikuti aturan yg kamu berikan
            def map_genre(g):
                if g in ("Aksi","Superhero","Komedi"): return 9
                if g in ("Drama","Sci-fi","Thriller"): return 7
                return 5  # documentary/art house

            def map_imdb(r):
                if r < 6.9: return 5
                if r < 7.5: return 7
                if r < 8.8: return 9
                return 10

            def map_age(age_label):
                if age_label == "SU": return 9
                if age_label == "13+": return 7
                if age_label == "17+": return 5
                return 3

            def map_dur(d):
                if d < 60: return 3
                if d <= 90: return 5
                if d <= 150: return 9
                return 7

            def map_year(y):
                if y > 2010: return 9
                if 2000 <= y <= 2010: return 7
                return 5

            # metadata alternatif (genre, imdb, age, duration, year)
            # saya isi sesuai film/populer — ini yang saya otomatiskan untuk kamu
            alt_meta = [
                ("Superhero", 8.4, "13+", 181, 2019),  # Avengers
                ("Sci-fi",   8.8, "13+", 148, 2010),  # Inception
                ("Superhero",9.0, "13+", 152, 2008),  # Dark Knight
                ("Sci-fi",   8.6, "13+", 169, 2014),  # Interstellar
                ("Drama",    8.6, "17+", 132, 2019),  # Parasite
                ("Komedi",   7.8, "SU",  100, 2019),  # Toy Story 4
                ("Drama",    8.5, "17+", 122, 2019),  # Joker
                ("Superhero",8.4, "13+", 117, 2023),  # Spider-Verse
                ("Drama",    7.7, "13+", 120, 2010),  # Social Network
                ("Drama",    8.6, "17+", 180, 2023),  # Oppenheimer
            ]

            raw_scores = np.zeros((n, NUM_CRITERIA), dtype=float)
            for i in range(n):
                g, imdb_v, age_v, dur_v, yr_v = alt_meta[i]
                raw_scores[i, 0] = map_genre(g)
                raw_scores[i, 1] = map_imdb(imdb_v)
                raw_scores[i, 2] = map_age(age_v)
                raw_scores[i, 3] = map_dur(dur_v)
                raw_scores[i, 4] = map_year(yr_v)

            # =========== buat pairwise matrix alternatif per kriteria dari raw local weights ===========
            alt_blocks = []
            alt_priority_cols = []  # akan menampung vektor bobot alternatif per kriteria
            for k in range(NUM_CRITERIA):
                col = raw_scores[:, k].astype(float)
                s = col.sum()
                if s == 0:
                    local_ws = np.ones(n) / n
                else:
                    local_ws = col / s  # bobot konsisten dari raw skor
                # pairwise konsisten (A_ij = w_i / w_j)
                Aw = pairwise_from_weights(local_ws)
                det = ahp_details(Aw)
                power = eigen_approx_from_matrix_power(Aw)
                # ambil eig_m4 sebagai bobot lokal alternatif (pastikan dinormalisasi)
                alt_vec = np.array(power["eig_m4"], dtype=float)
                if alt_vec.sum() != 0:
                    alt_vec = alt_vec / alt_vec.sum()
                else:
                    alt_vec = np.ones(n) / n

                block = {
                    "k": k,
                    "A": Aw.tolist(),
                    "col_sums": det["col_sums"],
                    "M2": power["M2"],
                    "M4": power["M4"],
                    "row_sums_m2": power["row_sums_m2"],
                    "row_sums_m4": power["row_sums_m4"],
                    "eig_m2": power["eig_m2"],
                    "eig_m4": power["eig_m4"],
                    "local_ws_from_raw": local_ws.tolist(),
                    "alt_priority": alt_vec.tolist(),
                }
                alt_blocks.append(block)
                alt_priority_cols.append(alt_vec)

            # bentuk matriks prioritas alternatif (n x NUM_CRITERIA)
            alt_priority_matrix = np.column_stack(alt_priority_cols)  # shape (10,5)

            # final synthesis: global score = alt_priority_matrix @ crit_weights_final
            crit_weights_final = np.array(crit_weights_final, dtype=float)
            global_scores = alt_priority_matrix.dot(crit_weights_final)

            ranking = ranking_from_scores(alternatives, global_scores.tolist())

            # update context untuk template
            context.update({
                "crit_matrix": Acrit.tolist(),
                "crit_power": crit_power,
                "crit_det": crit_det,
                "criteria_weights": crit_weights_final.tolist(),
                "criteria_lambda_max": crit_det["lambda_max"],
                "criteria_CI": crit_det["CI"],
                "criteria_CR": crit_det["CR"],
                "alt_blocks": alt_blocks,
                "raw_scores": raw_scores.tolist(),
                "alt_priority_matrix": alt_priority_matrix.tolist(),
                "global_scores": global_scores.tolist(),
                "ranking": ranking,
            })

        except Exception as e:
            context["error"] = str(e)

    return render_template("ahp.html", **context)

# ------------------------ AHP → WP route -----------------------------
@app.route("/ahp-wp", methods=["GET", "POST"])
def ahp_wp():
    context = {
        "error": None,
        "names": None,
        "matrix_input": None,
        "weights": None,
        "cost_flags": None,
        "exponents": None,
        "S": None,
        "V": None,
        "ranking": None,
        "criteria_names": ["C1", "C2", "C3", "C4", "C5"],
        "crit_matrix": None,
        "criteria_weights": None,
        "crit_det": None
    }

    # default matriks pairwise
    suggested_pairwise = np.array([
        [1,   3,   5,   3,   7],
        [1/3, 1,   3,   2,   5],
        [1/5, 1/3, 1, 0.5,   3],
        [1/3, 0.5, 2,   1,   3],
        [1/7, 1/5, 1/3, 1/3, 1]
    ], dtype=float)
    context["default_pairwise"] = suggested_pairwise.tolist()

    if request.method == "POST":
        try:
            # 1️⃣ baca matriks pairwise AHP
            Acrit = np.zeros((NUM_CRITERIA, NUM_CRITERIA), dtype=float)
            for i in range(NUM_CRITERIA):
                for j in range(NUM_CRITERIA):
                    Acrit[i, j] = safe_float(request.form.get(f"c_{i}_{j}", ""), 0.0)

            # perbaiki diagonal & reciprocal
            for i in range(NUM_CRITERIA):
                for j in range(NUM_CRITERIA):
                    if i == j:
                        Acrit[i, j] = 1.0
                    elif Acrit[i, j] == 0 and Acrit[j, i] != 0:
                        Acrit[i, j] = 1.0 / Acrit[j, i]
                    elif Acrit[i, j] == 0:
                        Acrit[i, j] = 1.0

            # 2️⃣ hitung bobot AHP
            crit_power = eigen_approx_from_matrix_power(Acrit)
            crit_det = ahp_details(Acrit)
            crit_eig_m4 = np.array(crit_power["eig_m4"], dtype=float)
            weights = normalize_weights_to_1(crit_eig_m4.tolist())

            # 3️⃣ ambil data film
            kandidat = int(request.form.get("kandidat", "0"))
            if kandidat < 1:
                raise ValueError("Jumlah kandidat minimal 1.")

            names, matrix = [], []
            for i in range(1, kandidat + 1):
                names.append(request.form.get(f"nama_{i}", f"Kandidat {i}"))
                row = [safe_float(request.form.get(f"film_{i}_c{j}", "0")) for j in range(1, NUM_CRITERIA + 1)]
                matrix.append(row)

            # 4️⃣ cost flags (default 0 semua)
            cfraw = [s for s in request.form.get("cost_flags", "0,0,0,0,0").split(",") if s.strip() != ""]
            cost_flags = [int(float(x)) for x in cfraw]
            # pad cost_flags jika kurang
            if len(cost_flags) < NUM_CRITERIA:
                cost_flags += [0] * (NUM_CRITERIA - len(cost_flags))
            else:
                cost_flags = cost_flags[:NUM_CRITERIA]

            # 5️⃣ jalankan WP dengan bobot AHP
            wp_res = compute_wp(names, matrix, weights, cost_flags)

            # 6️⃣ update context
            context.update({
                "crit_matrix": Acrit.tolist(),
                "criteria_weights": weights,
                "crit_det": crit_det,
                "names": names,
                "matrix_input": [{"name": names[i], "vals": matrix[i]} for i in range(len(names))],
                "weights": weights,
                "cost_flags": cost_flags,
                "exponents": wp_res["exponents"],
                "S": wp_res["S"],
                "V": wp_res["V"],
                "ranking": wp_res["ranking"],
            })
        except Exception as e:
            context["error"] = str(e)

    return render_template("ahp_wp.html", **context)


# ------------------------ AHP → SAW (pairwise input) -----------------------------
@app.route("/ahp-saw", methods=["GET", "POST"])
def ahp_saw():
    criteria_names = ["C1", "C2", "C3", "C4", "C5"]

    # default matriks pairwise seperti contoh
    suggested_pairwise = np.array([
        [1,   3,   5,   3,   7],
        [1/3, 1,   3,   2,   5],
        [1/5, 1/3, 1, 0.5,   3],
        [1/3, 0.5, 2,   1,   3],
        [1/7, 1/5, 1/3, 1/3, 1]
    ], dtype=float)

    context = {
        "error": None,
        "criteria_names": criteria_names,
        "default_pairwise": suggested_pairwise.tolist(),
        "weights": None,
        "crit_det": None,
        "matrix_input": None,
        "max_vals": None,
        "norm": None,
        "weighted": None,
        "scores": None,
        "ranking": None,
        "names": None,
    }

    if request.method == "POST":
        try:
            # 1) Baca matriks pairwise
            Acrit = np.zeros((NUM_CRITERIA, NUM_CRITERIA), dtype=float)
            for i in range(NUM_CRITERIA):
                for j in range(NUM_CRITERIA):
                    Acrit[i, j] = safe_float(request.form.get(f"c_{i}_{j}", ""), 0.0)

            # rapikan diagonal & resiprokal
            for i in range(NUM_CRITERIA):
                for j in range(NUM_CRITERIA):
                    if i == j:
                        Acrit[i, j] = 1.0
                    elif Acrit[i, j] == 0 and Acrit[j, i] != 0:
                        Acrit[i, j] = 1.0 / Acrit[j, i]
                    elif Acrit[i, j] == 0:
                        Acrit[i, j] = 1.0

            # 2) Hitung bobot AHP
            crit_power = eigen_approx_from_matrix_power(Acrit)
            crit_det = ahp_details(Acrit)
            eig_m4 = np.array(crit_power["eig_m4"], dtype=float)
            weights_ahp = normalize_weights_to_1(eig_m4)

            # 3) Ambil data film dari form
            kandidat = int(request.form.get("kandidat", "0"))
            if kandidat < 1:
                raise ValueError("Jumlah film minimal 1.")

            names, matrix = [], []
            for i in range(1, kandidat + 1):
                name = (request.form.get(f"nama_{i}", "") or f"Film {i}").strip()
                names.append(name)
                # nilai tiap kriteria film
                row = [safe_float(request.form.get(f"film_{i}_c{j}", "0")) for j in range(1, NUM_CRITERIA + 1)]
                matrix.append(row)

            # 4) Jalankan SAW dengan bobot dari AHP
            res = compute_saw(names, matrix, weights_ahp)

            # 5) Masukkan ke context
            context.update({
                "weights": weights_ahp,
                "crit_det": crit_det,
                "names": names,
                "matrix_input": [{"name": names[i], "vals": matrix[i]} for i in range(len(names))],
                "max_vals": res["max_vals"],
                "norm": [{"name": names[i], "vals": res["norm"][i]} for i in range(len(names))],
                "weighted": [{"name": names[i], "vals": res["weighted"][i], "vi": res["scores"][i]} for i in range(len(names))],
                "scores": res["scores"],
                "ranking": res["ranking"],
            })
        except Exception as e:
            context["error"] = str(e)

    return render_template("ahp_saw.html", **context)

# ======================== Runner =========================
if __name__ == "__main__":
    app.run(debug=True)  

