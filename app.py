from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. 경로 및 기본 설정
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "train_u6lujuX_CVtuZ9i.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
META_PATH = MODEL_DIR / "model_meta.joblib"

st.set_page_config(
    page_title="대출 승인 예측 및 데이터 분석",
    page_icon="🏦",
    layout="wide",
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(
    style="whitegrid",
    rc={"font.family": "Malgun Gothic"},
)

# 변수명 한국어 변환
FEATURE_LABELS = {
    "ApplicantIncome": "신청자 소득",
    "CoapplicantIncome": "공동 신청자 소득",
    "LoanAmount": "대출 금액",
    "Loan_Amount_Term": "대출 기간",
    "Credit_History": "신용 이력",
    "Gender": "성별",
    "Married": "혼인 여부",
    "Dependents": "부양가족 수",
    "Education": "교육 수준",
    "Self_Employed": "자영업 여부",
    "Property_Area": "부동산 지역",
    "Loan_Status": "대출 승인 여부",
    "Loan_ID": "대출 신청 식별번호",
}

CATEGORY_LABELS = {
    "Female": "여성",
    "Male": "남성",
    "Yes": "예",
    "No": "아니오",
    "Graduate": "대졸 이상",
    "Not Graduate": "비대졸",
    "Rural": "농촌",
    "Semiurban": "준도시",
    "Urban": "도시",
    "3+": "3명 이상",
    "Y": "승인",
    "N": "거절",
    "0": "0명",
    "1": "1명",
    "2": "2명",
    "0.0": "불량 또는 미확인",
    "1.0": "양호",
    "결측": "결측값",
    "nan": "결측값",
    "Unknown": "미확인",
}

TARGET_LABELS = {
    "Y": "승인",
    "N": "거절",
}


def korean_feature_name(name):
    """원본 변수명과 원-핫 인코딩 변수명을 한국어로 변환한다."""
    name = str(name)

    for original, korean in FEATURE_LABELS.items():
        if name == original:
            return korean

        if name.startswith(original + "_"):
            suffix = name[len(original) + 1 :]
            return f"{korean} - " f"{CATEGORY_LABELS.get(suffix, suffix)}"

    return CATEGORY_LABELS.get(name, name)


# -----------------------------------------------------------------------------
# 2. 모델 및 데이터 불러오기
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="저장된 모델을 불러오는 중입니다...")
def load_artifacts():
    """학습된 모델과 메타데이터를 불러온다."""
    if not MODEL_PATH.exists():
        return (
            None,
            {},
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}",
        )

    try:
        loaded_model = joblib.load(MODEL_PATH)

        loaded_meta = joblib.load(META_PATH) if META_PATH.exists() else {}

        return loaded_model, loaded_meta, None

    except Exception as exc:
        return None, {}, f"모델 파일을 읽지 못했습니다: {exc}"


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data():
    """학습용 CSV 데이터를 불러온다."""
    if not DATA_PATH.exists():
        return None, f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}"

    try:
        return pd.read_csv(DATA_PATH), None

    except Exception as exc:
        return None, f"데이터 파일을 읽지 못했습니다: {exc}"


model, meta, model_error = load_artifacts()
df, data_error = load_data()


# -----------------------------------------------------------------------------
# 3. 공통 화면 및 데이터 확인
# -----------------------------------------------------------------------------
st.title("🏦 대출 승인 예측 및 데이터 분석")

st.caption(
    "데이터 탐색, 상관관계 및 다중공선성 확인, "
    "모델 평가, 개별 신청 정보 예측을 제공합니다."
)

st.warning(
    "교육용 프로토타입입니다. 이 모델은 실제 금융 심사에 대한 "
    "검증 및 공정성 평가를 거치지 않았으며, "
    "실제 대출 승인 여부를 결정하는 데 사용하면 안 됩니다."
)

if model_error:
    st.error(model_error)
    st.info(
        "먼저 loan_approval.ipynb를 위에서 아래로 실행해 "
        "model/model.pkl과 model/model_meta.joblib을 생성해 주세요."
    )
    st.stop()

if data_error:
    st.error(data_error)
    st.stop()

if "Loan_Status" not in df.columns:
    st.error(
        "데이터에 Loan_Status 타깃 변수가 없습니다. " "학습용 CSV 파일을 확인해 주세요."
    )
    st.stop()


# 모델 메타데이터에 저장된 입력 변수 사용
feature_columns = meta.get(
    "feature_columns",
    [c for c in df.columns if c not in ["Loan_ID", "Loan_Status"]],
)

numeric_features = meta.get(
    "numeric_features",
    df[feature_columns].select_dtypes(include=np.number).columns.tolist(),
)

numeric_features = [c for c in numeric_features if c in feature_columns]

categorical_features = [c for c in feature_columns if c not in numeric_features]

# 중요:
# 학습 노트북에서 저장한 모델이 전처리 Pipeline이라면
# 앱에서 별도로 인코딩이나 표준화를 수행하지 않는다.
# 원본 변수명과 입력값을 유지한 DataFrame을 모델에 전달한다.


# -----------------------------------------------------------------------------
# 4. 탭 구성
# -----------------------------------------------------------------------------
tab_pred, tab_eda, tab_ml, tab_trouble = st.tabs(
    [
        "🔎 신청 정보 예측",
        "📊 데이터 탐색 및 인사이트",
        "🧠 머신러닝 분석",
        "🛠 트러블슈팅",
    ]
)


# =============================================================================
# TAB 1. 개별 신청 정보 예측
# =============================================================================
with tab_pred:
    st.subheader("신청 정보 입력")

    st.write(
        "학습 시 저장된 전처리 Pipeline을 사용해 " "입력값을 변환한 뒤 예측합니다."
    )

    with st.form("prediction_form", clear_on_submit=False):
        cols = st.columns(2)
        inputs = {}

        for i, col in enumerate(feature_columns):
            target_col = cols[i % 2]
            label = f"{FEATURE_LABELS.get(col, col)} ({col})"

            # 수치형 변수 입력
            if col in numeric_features:
                values = pd.to_numeric(df[col], errors="coerce").dropna()

                median = float(values.median()) if not values.empty else 0.0

                lower = float(values.min()) if not values.empty else 0.0

                upper = (
                    float(values.max()) if not values.empty else max(median + 1, 1.0)
                )

                if upper <= lower:
                    upper = lower + 1.0

                # 신용 이력은 선택형 입력으로 제공
                if col == "Credit_History":
                    options = sorted(
                        pd.to_numeric(
                            values,
                            errors="coerce",
                        )
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    options = options or [0.0, 1.0]

                    default = min(
                        options,
                        key=lambda v: abs(float(v) - median),
                    )

                    inputs[col] = target_col.selectbox(
                        label,
                        options=options,
                        index=options.index(default),
                        format_func=lambda value: (
                            "양호 (1)" if float(value) == 1 else "불량 또는 미확인 (0)"
                        ),
                    )

                else:
                    integer_like = (
                        bool(
                            np.all(
                                np.isclose(
                                    values.to_numpy() % 1,
                                    0,
                                )
                            )
                        )
                        if not values.empty
                        else True
                    )

                    step = 1.0 if integer_like else 0.1
                    default = float(np.clip(median, lower, upper))

                    inputs[col] = target_col.number_input(
                        label,
                        min_value=float(lower),
                        max_value=float(upper),
                        value=default,
                        step=step,
                        format=("%.0f" if integer_like else "%.2f"),
                    )

            # 범주형 변수 입력
            else:
                options = sorted(df[col].dropna().astype(str).unique().tolist())

                if not options:
                    options = ["Unknown"]

                if col == "Dependents":
                    options = sorted(
                        options,
                        key=lambda v: (
                            v == "3+",
                            int(v) if v.isdigit() else 999,
                        ),
                    )

                preferred = {
                    "Education": "Graduate",
                    "Married": "Yes",
                    "Property_Area": "Semiurban",
                    "Gender": "Male",
                    "Self_Employed": "No",
                }.get(col)

                default = preferred if preferred in options else options[0]

                inputs[col] = target_col.selectbox(
                    label,
                    options,
                    index=options.index(default),
                )

        submitted = st.form_submit_button(
            "대출 승인 여부 예측",
            type="primary",
            use_container_width=True,
        )

    # 예측 실행
    if submitted:
        try:
            input_df = pd.DataFrame(
                [{c: inputs[c] for c in feature_columns}],
                columns=feature_columns,
            )

            prediction = int(model.predict(input_df)[0])

            if not hasattr(model, "predict_proba"):
                st.error("현재 모델에서 예측 확률을 제공하지 않습니다.")
                st.stop()

            probabilities = model.predict_proba(input_df)[0]

            # Pipeline의 마지막 분류기에서 클래스 순서를 확인
            classifier = (
                model.named_steps.get("classifier")
                if hasattr(model, "named_steps")
                else model
            )

            classes = list(getattr(classifier, "classes_", [0, 1]))

            if 1 in classes:
                approved_index = classes.index(1)
            else:
                approved_index = int(np.argmax(classes))

            approved_probability = float(probabilities[approved_index])

            left, right = st.columns([1, 1])

            with left:
                if prediction == 1:
                    st.success("모델 분류 결과: 승인 그룹 (Y)")
                else:
                    st.error("모델 분류 결과: 거절 그룹 (N)")

                st.metric(
                    "대출 승인 예측 확률",
                    f"{approved_probability:.1%}",
                )

                st.progress(
                    float(
                        np.clip(
                            approved_probability,
                            0,
                            1,
                        )
                    )
                )

                if approved_probability >= 0.5:
                    st.caption(
                        "이 모델의 분류 기준에서 승인 확률이 "
                        "상대적으로 높게 추정되었습니다."
                    )
                else:
                    st.caption(
                        "이 모델의 분류 기준에서 승인 확률이 "
                        "상대적으로 낮게 추정되었습니다."
                    )

            with right:
                st.markdown("**입력한 신청 정보**")

                display_input = input_df.T.rename(columns={0: "입력값"})

                display_input.index = [
                    FEATURE_LABELS.get(c, c) for c in display_input.index
                ]

                display_input["입력값"] = display_input["입력값"].replace(
                    CATEGORY_LABELS
                )

                st.dataframe(
                    display_input,
                    use_container_width=True,
                )

            st.caption(
                "확률은 학습 데이터에서 패턴을 학습한 모델의 "
                "추정값이며, 실제 금융기관의 판단이나 "
                "승인 결과를 보장하지 않습니다."
            )

        except Exception as exc:
            st.error(f"예측 중 오류가 발생했습니다: {exc}")

            st.info(
                "노트북에서 생성한 모델 파일과 앱의 "
                "feature_columns 구성이 일치하는지 확인해 주세요."
            )

    else:
        st.info("신청 정보를 설정한 뒤 " "**대출 승인 여부 예측** 버튼을 눌러 주세요.")


# =============================================================================
# TAB 2. 데이터 탐색 및 인사이트 대시보드
# =============================================================================
with tab_eda:
    st.subheader("데이터 구성 및 품질")

    total = len(df)

    approved_count = int((df["Loan_Status"] == "Y").sum())

    rejected_count = int((df["Loan_Status"] == "N").sum())

    approval_rate = approved_count / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("전체 신청 건수", f"{total:,}")
    c2.metric("전체 변수 수", f"{df.shape[1]}")
    c3.metric("승인 비율", f"{approval_rate:.1f}%")
    c4.metric(
        "결측값 셀 수",
        f"{int(df.isna().sum().sum()):,}",
    )

    # 원본 데이터 미리보기
    with st.expander("원본 데이터 미리보기"):
        preview = df.head(20).rename(columns=FEATURE_LABELS).replace(CATEGORY_LABELS)

        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True,
        )

    # 변수별 데이터 품질
    st.markdown("#### 변수별 데이터 품질")

    missing = pd.DataFrame(
        {
            "변수": [FEATURE_LABELS.get(c, c) for c in df.columns],
            "자료형": df.dtypes.astype(str).values,
            "결측 수": df.isna().sum().values,
            "결측 비율(%)": (df.isna().mean().values * 100).round(2),
            "고유값 수": df.nunique(dropna=True).values,
        }
    ).sort_values(
        "결측 비율(%)",
        ascending=False,
    )

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True,
    )

    # 승인·거절 분포와 결측치 비율
    chart_a, chart_b = st.columns(2)

    with chart_a:
        st.markdown("#### 승인·거절 데이터 분포")

        status_counts = pd.Series(
            {
                "승인": approved_count,
                "거절": rejected_count,
            }
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            status_counts.index,
            status_counts.values,
        )

        ax.set_ylabel("신청 건수")
        ax.set_title("대출 승인 여부 분포")

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with chart_b:
        st.markdown("#### 결측치 비율")

        missing_nonzero = missing[missing["결측 수"] > 0].sort_values("결측 비율(%)")

        if not missing_nonzero.empty:
            fig, ax = plt.subplots(figsize=(7, 4))

            ax.barh(
                missing_nonzero["변수"],
                missing_nonzero["결측 비율(%)"],
            )

            ax.set_xlabel("결측 비율 (%)")
            ax.set_title("변수별 결측치 비율")

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        else:
            st.success("결측치가 없습니다.")

    # 상관관계 히트맵
    st.markdown("#### 변수 간 상관관계 히트맵")

    st.caption(
        "범주형 변수는 원-핫 인코딩하고, 타깃 변수는 "
        "거절=0·승인=1로 변환했습니다. "
        "상관관계는 연관성을 나타내며 인과관계를 의미하지 않습니다."
    )

    corr_base = df.drop(
        columns=["Loan_ID"],
        errors="ignore",
    ).copy()

    corr_base["Loan_Status"] = corr_base["Loan_Status"].map({"N": 0, "Y": 1})

    for col in corr_base.columns:
        if corr_base[col].dtype == "object":
            corr_base[col] = corr_base[col].fillna("결측").astype(str)
        else:
            corr_base[col] = pd.to_numeric(
                corr_base[col],
                errors="coerce",
            )

            corr_base[col] = corr_base[col].fillna(corr_base[col].median())

    categorical_corr_cols = [
        c for c in corr_base.select_dtypes(exclude=np.number).columns
    ]

    corr_encoded = pd.get_dummies(
        corr_base,
        columns=categorical_corr_cols,
        dtype=float,
    )

    corr_encoded = corr_encoded.rename(
        columns={c: korean_feature_name(c) for c in corr_encoded.columns}
    )

    corr_matrix = corr_encoded.corr(numeric_only=True)

    show_all_corr = st.checkbox(
        "모든 변수 표시 (히트맵이 커질 수 있음)",
        value=False,
    )

    if not show_all_corr and corr_matrix.shape[0] > 18:
        target_name = korean_feature_name("Loan_Status")

        if target_name in corr_matrix.columns:
            strongest = (
                corr_matrix[target_name]
                .drop(labels=[target_name], errors="ignore")
                .abs()
                .sort_values(ascending=False)
                .head(17)
                .index.tolist()
            )

            display_cols = strongest + [target_name]

        else:
            display_cols = (
                corr_matrix.abs()
                .mean()
                .sort_values(ascending=False)
                .head(18)
                .index.tolist()
            )

        corr_to_plot = corr_matrix.loc[
            display_cols,
            display_cols,
        ]

    else:
        corr_to_plot = corr_matrix

    fig_w = max(
        9,
        min(18, corr_to_plot.shape[1] * 0.72),
    )

    fig_h = max(
        7,
        min(16, corr_to_plot.shape[0] * 0.62),
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        corr_to_plot,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.25,
        ax=ax,
    )

    ax.set_title("변수 간 상관관계 (Pearson correlation)")

    ax.tick_params(
        axis="x",
        rotation=70,
        labelsize=8,
    )

    ax.tick_params(
        axis="y",
        rotation=0,
        labelsize=8,
    )

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    # VIF 분석
    st.markdown("#### 다중공선성 확인 (VIF)")

    st.caption(
        "VIF는 수치형 설명 변수 사이의 선형 중복 정도를 "
        "확인하는 참고 지표입니다. "
        "범주형 변수 전체의 다중공선성을 평가하는 값은 아닙니다."
    )

    vif_cols = [
        c for c in numeric_features if c in df.columns and c != "Credit_History"
    ]

    if len(vif_cols) >= 2:
        vif_data = df[vif_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        vif_data = vif_data.fillna(vif_data.median()).fillna(0)

        # 상수 열은 VIF 계산에서 제외
        vif_cols = [
            c for c in vif_data.columns if vif_data[c].nunique(dropna=False) > 1
        ]

        vif_data = vif_data[vif_cols]

        if len(vif_cols) >= 2:
            try:
                from statsmodels.stats.outliers_influence import (
                    variance_inflation_factor,
                )

                from statsmodels.tools.tools import add_constant

                design = add_constant(
                    vif_data.astype(float),
                    has_constant="add",
                )

                vif_values = []

                for i, col in enumerate(design.columns):
                    if col == "const":
                        continue

                    try:
                        value = float(
                            variance_inflation_factor(
                                design.to_numpy(),
                                i,
                            )
                        )

                    except Exception:
                        value = np.nan

                    vif_values.append(
                        {
                            "변수": FEATURE_LABELS.get(col, col),
                            "VIF": value,
                        }
                    )

                vif_df = pd.DataFrame(vif_values).sort_values(
                    "VIF",
                    ascending=False,
                    na_position="last",
                )

                st.dataframe(
                    vif_df.round(3),
                    use_container_width=True,
                    hide_index=True,
                )

                finite_vif = vif_df[np.isfinite(vif_df["VIF"])]

                if not finite_vif.empty:
                    high_vif = finite_vif[finite_vif["VIF"] >= 5]

                    if not high_vif.empty:
                        st.warning(
                            "VIF가 5 이상인 변수가 있습니다. "
                            "변수 간 중복 정보 가능성을 검토하되, "
                            "자동으로 변수를 제거하지 말고 "
                            "모델 성능과 해석 목적을 함께 고려하세요."
                        )

                    else:
                        st.success(
                            "계산 가능한 수치형 변수에서 "
                            "VIF 5 이상인 변수는 확인되지 않았습니다."
                        )

            except ImportError:
                st.info(
                    "VIF 표를 보려면 statsmodels를 설치해 주세요: "
                    "pip install statsmodels"
                )

        else:
            st.info(
                "VIF를 계산할 수 있는 변동성이 있는 " "수치형 변수가 충분하지 않습니다."
            )

    else:
        st.info("VIF 계산에 필요한 수치형 변수가 충분하지 않습니다.")

    # 수치형 변수 분포
    st.markdown("#### 수치형 변수 분포")

    numeric_plot_cols = [
        c for c in df.select_dtypes(include=np.number).columns if c != "Credit_History"
    ]

    if numeric_plot_cols:
        selected_num = st.selectbox(
            "분포를 확인할 변수",
            numeric_plot_cols,
            index=(
                numeric_plot_cols.index("ApplicantIncome")
                if "ApplicantIncome" in numeric_plot_cols
                else 0
            ),
            format_func=lambda c: FEATURE_LABELS.get(c, c),
        )

        hist_df = df.copy()

        hist_df["승인 여부"] = hist_df["Loan_Status"].map(TARGET_LABELS)

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.histplot(
            data=hist_df,
            x=selected_num,
            hue="승인 여부",
            bins=25,
            element="step",
            common_norm=False,
            ax=ax,
        )

        ax.set_title(f"{FEATURE_LABELS.get(selected_num, selected_num)} 분포")

        ax.set_xlabel(FEATURE_LABELS.get(selected_num, selected_num))

        ax.set_ylabel("신청 건수")

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    # 범주별 승인 비율
    st.markdown("#### 변수별 승인 비율")

    categorical_cols = [
        c
        for c in df.select_dtypes(exclude=np.number).columns
        if c not in ["Loan_ID", "Loan_Status"]
    ]

    relation_cols = categorical_cols + (
        ["Credit_History"] if "Credit_History" in df.columns else []
    )

    if relation_cols:
        selected_cat = st.selectbox(
            "승인 비율을 비교할 변수",
            relation_cols,
            key="approval_rate_var",
            format_func=lambda c: FEATURE_LABELS.get(c, c),
        )

        temp = df[[selected_cat, "Loan_Status"]].copy()

        temp[selected_cat] = temp[selected_cat].fillna("결측").astype(str)

        rate = (
            temp.groupby(
                selected_cat,
                dropna=False,
            )["Loan_Status"]
            .apply(lambda values: (values == "Y").mean() * 100)
            .sort_values(ascending=False)
            .reset_index(name="승인 비율(%)")
        )

        rate_display = rate.rename(columns={selected_cat: "범주"}).copy()

        rate_display["범주"] = rate_display["범주"].map(
            lambda x: CATEGORY_LABELS.get(
                str(x),
                str(x),
            )
        )

        st.dataframe(
            rate_display.round(2),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            rate_display["범주"],
            rate["승인 비율(%)"],
        )

        ax.set_title(f"{FEATURE_LABELS.get(selected_cat, selected_cat)}별 승인 비율")

        ax.set_ylabel("승인 비율 (%)")
        ax.tick_params(axis="x", rotation=20)

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


# =============================================================================
# TAB 3. 모델 비교 및 테스트 성능
# =============================================================================
with tab_ml:
    st.subheader("모델 비교 및 성능 평가")

    st.write("교차검증 결과와 별도 테스트 데이터의 성능을 " "구분해 확인합니다.")

    # 모델 비교표
    comparison = meta.get("model_comparison", [])

    if comparison:
        comp = pd.DataFrame(comparison)

        rename_map = {
            "모델": "모델명",
            "CV Accuracy 평균": "교차검증 정확도 평균",
            "CV Precision 평균": "교차검증 정밀도 평균",
            "CV Recall 평균": "교차검증 재현율 평균",
            "CV F1 평균": "교차검증 F1 평균",
            "CV ROC-AUC 평균": "교차검증 ROC-AUC 평균",
            "CV F1 표준편차": "교차검증 F1 표준편차",
        }

        st.dataframe(
            comp.rename(columns=rename_map).round(4),
            use_container_width=True,
            hide_index=True,
        )

        metric_options = [
            c
            for c in [
                "CV F1 평균",
                "CV ROC-AUC 평균",
                "CV Accuracy 평균",
            ]
            if c in comp.columns
        ]

        if metric_options and "모델" in comp.columns:
            selected_metric = st.selectbox(
                "모델 비교 기준",
                metric_options,
                format_func=lambda x: {
                    "CV F1 평균": "교차검증 F1 평균",
                    "CV ROC-AUC 평균": "교차검증 ROC-AUC 평균",
                    "CV Accuracy 평균": "교차검증 정확도 평균",
                }.get(x, x),
            )

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.bar(
                comp["모델"],
                comp[selected_metric],
            )

            ax.set_ylim(0, 1)
            ax.set_ylabel("점수")
            ax.tick_params(axis="x", rotation=15)
            ax.set_title("교차검증 기준 모델 비교")

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    else:
        st.info(
            "모델 비교 결과가 없습니다. "
            "노트북을 다시 실행해 메타데이터를 저장해 주세요."
        )

    # 테스트 데이터 성능
    metrics = meta.get("metrics", {})

    if metrics:
        st.markdown("#### 최종 테스트 데이터 성능")

        metric_order = [
            name
            for name in [
                "정확도",
                "정밀도",
                "재현율",
                "F1 점수",
                "ROC-AUC",
            ]
            if name in metrics
        ]

        if metric_order:
            metric_cols = st.columns(len(metric_order))

            for col, name in zip(
                metric_cols,
                metric_order,
            ):
                col.metric(
                    name,
                    f"{float(metrics[name]):.3f}",
                )

        st.caption(
            "테스트 데이터 결과는 모델 학습 및 후보 선택 과정에서 "
            "분리한 데이터의 평가 결과입니다."
        )

    # 혼동 행렬
    confusion = meta.get("confusion_matrix")

    if confusion:
        st.markdown("#### 혼동 행렬")

        cm_df = pd.DataFrame(
            confusion,
            index=["실제 거절", "실제 승인"],
            columns=["예측 거절", "예측 승인"],
        )

        st.dataframe(
            cm_df,
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
        )

        ax.set_xlabel("예측 결과")
        ax.set_ylabel("실제 결과")

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    # ROC 곡선
    roc_data = meta.get("roc_curve", {})

    fpr = roc_data.get("fpr", [])
    tpr = roc_data.get("tpr", [])

    if len(fpr) and len(tpr):
        st.markdown("#### ROC 곡선")

        fig, ax = plt.subplots(figsize=(7, 5))

        auc_value = metrics.get("ROC-AUC")

        label = (
            f"모델 ROC 곡선 (AUC={float(auc_value):.3f})"
            if auc_value is not None
            else "모델 ROC 곡선"
        )

        ax.plot(
            fpr,
            tpr,
            label=label,
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="무작위 분류 기준선",
        )

        ax.set_xlabel("거짓 양성률")
        ax.set_ylabel("참 양성률")
        ax.set_title("ROC 곡선")
        ax.legend(loc="lower right")

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    else:
        st.info(
            "ROC 곡선 정보가 없습니다. "
            "노트북을 다시 실행해 메타데이터를 갱신해 주세요."
        )

    # 순열 기반 변수 중요도
    importance = meta.get("permutation_importance", [])

    if importance:
        st.markdown("#### 순열 기반 변수 중요도")

        importance_df = pd.DataFrame(importance)

        if {
            "feature",
            "importance_mean",
        }.issubset(importance_df.columns):

            importance_df = importance_df.sort_values(
                "importance_mean",
                ascending=True,
            ).tail(12)

            importance_df["변수명"] = importance_df["feature"].map(
                lambda x: FEATURE_LABELS.get(x, x)
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            std = (
                importance_df["importance_std"]
                if "importance_std" in importance_df.columns
                else None
            )

            ax.barh(
                importance_df["변수명"],
                importance_df["importance_mean"],
                xerr=std,
            )

            ax.set_title("변수 값을 무작위로 섞었을 때의 성능 변화")

            ax.set_xlabel("평균 F1 점수 감소량")

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    # 결과 해석 시 유의사항
    st.markdown("#### 결과 해석 시 유의사항")

    st.markdown(
        "- 데이터는 614건으로 규모가 크지 않아 "
        "표본 구성에 따라 평가 결과가 달라질 수 있습니다.\n"
        "- 관찰된 상관관계나 변수 중요도는 "
        "인과관계를 의미하지 않습니다.\n"
        "- 정확도 하나만 보지 말고 정밀도·재현율·F1·ROC-AUC를 "
        "함께 살펴야 합니다.\n"
        "- 실제 적용 전에는 외부 데이터 검증, 공정성 평가, "
        "임계값 검토 및 관련 규정 검토가 필요합니다."
    )


# =============================================================================
# TAB 4. 트러블슈팅 및 구현 시 고려 사항
# =============================================================================
with tab_trouble:
    st.subheader("구현 과정의 주요 문제와 해결 방법")

    st.write(
        "학습·예측 파이프라인, 결측치 처리, 모델 비교 기준을 "
        "일관되게 관리하기 위한 내용을 정리합니다."
    )

    st.caption(
        "아래 내용은 프로젝트에서 적용한 설계와 해결 방법을 "
        "정리한 것입니다. 별도 오류 로그가 없는 항목은 "
        "실제 발생 오류가 아니라 예방 및 대응 방안으로 해석해야 합니다."
    )

    # -------------------------------------------------------------------------
    # 트러블슈팅 1. 모델과 전처리 연동
    # -------------------------------------------------------------------------
    with st.expander(
        "① 모델과 전처리 연동",
        expanded=True,
    ):
        st.markdown(
            "**문제 상황**\n\n"
            "학습 데이터와 예측 입력 데이터의 전처리 방식 또는 "
            "열 구성이 달라지면 예측 오류나 일관성 저하가 발생할 수 있습니다."
        )

        st.markdown(
            "**해결 방법**\n\n"
            "수치형 변수의 결측치 대체·표준화와 "
            "범주형 변수의 결측치 대체·원-핫 인코딩을 "
            "분류 모델과 하나의 Pipeline으로 구성했습니다. "
            "앱은 원본 열 이름과 값을 유지한 DataFrame을 "
            "`model.predict()`에 전달합니다."
        )

        pipeline_ok = hasattr(model, "named_steps") and any(
            hasattr(step, "transform") for step in model.named_steps.values()
        )

        if pipeline_ok:
            st.success(
                "현재 저장된 모델에서 전처리 단계를 포함한 "
                "Pipeline 구조를 확인했습니다."
            )
        else:
            st.warning(
                "저장된 모델이 전처리 Pipeline인지 자동 확인하지 못했습니다. "
                "노트북에서 저장한 model.pkl 구성을 확인해 주세요."
            )

        st.markdown(
            "**적용 결과**\n\n"
            "학습과 예측에서 동일한 전처리 규칙을 사용하도록 하여 "
            "입력 형식의 불일치 가능성을 줄였습니다."
        )

    # -------------------------------------------------------------------------
    # 트러블슈팅 2. 결측치 처리
    # -------------------------------------------------------------------------
    with st.expander("② 결측치 처리"):
        st.markdown(
            "**문제 상황**\n\n"
            "원본 데이터에는 일부 변수에 결측값이 존재하므로, "
            "결측치를 처리하지 않고 학습하면 일부 알고리즘에서 "
            "오류가 발생할 수 있습니다."
        )

        st.markdown(
            "**해결 방법**\n\n"
            "수치형 변수는 중앙값으로 대체하고 표준화했으며, "
            "범주형 변수는 최빈값으로 대체한 뒤 "
            "원-핫 인코딩하도록 구성했습니다."
        )

        missing_cells = int(df.isna().sum().sum())

        st.metric(
            "원본 데이터의 결측 셀 수",
            f"{missing_cells:,}개",
        )

        st.markdown(
            "**적용 결과**\n\n"
            "전처리 Pipeline 안에서 결측값을 처리하므로, "
            "앱 입력 데이터에도 학습 시와 같은 "
            "결측치 처리 규칙이 적용됩니다."
        )

    # -------------------------------------------------------------------------
    # 트러블슈팅 3. 모델 성능 비교
    # -------------------------------------------------------------------------
    with st.expander("③ 모델 성능 비교"):
        st.markdown(
            "**문제 상황**\n\n"
            "단일 모델이나 정확도만으로 모델을 선택하면 "
            "클래스 분포의 영향을 충분히 고려하지 못할 수 있습니다."
        )

        st.markdown(
            "**해결 방법**\n\n"
            "Dummy Classifier, Logistic Regression, Decision Tree, "
            "Random Forest, Extra Trees를 동일한 5-Fold Stratified "
            "교차검증 기준으로 비교하고, 평균 F1-score를 "
            "주요 선정 기준으로 사용했습니다."
        )

        comparison_records = meta.get(
            "model_comparison",
            [],
        )

        if comparison_records:
            trouble_comp = pd.DataFrame(comparison_records)

            if "CV F1 평균" in trouble_comp.columns:
                trouble_comp = trouble_comp.sort_values(
                    "CV F1 평균",
                    ascending=False,
                )

                st.dataframe(
                    trouble_comp.rename(
                        columns={
                            "모델": "모델명",
                            "CV F1 평균": "교차검증 F1 평균",
                        }
                    ).round(4),
                    use_container_width=True,
                    hide_index=True,
                )

                best_row = trouble_comp.iloc[0]

                st.success(
                    "저장된 비교 결과에서 평균 F1-score가 "
                    f"가장 높은 모델: "
                    f"{best_row.get('모델', '확인 불가')} "
                    f"({float(best_row['CV F1 평균']):.4f})"
                )

            else:
                st.info(
                    "모델 비교 메타데이터에 'CV F1 평균' 항목이 없습니다. "
                    "노트북 저장 결과를 확인해 주세요."
                )

        else:
            st.info(
                "모델 비교 결과가 없습니다. "
                "노트북을 위에서 아래로 실행해 "
                "메타데이터를 생성해 주세요."
            )

        st.markdown(
            "**적용 결과**\n\n"
            "동일한 교차검증 기준으로 모델을 비교해 "
            "특정 모델의 성능을 일관된 기준에서 평가했습니다."
        )

    # -------------------------------------------------------------------------
    # 실행 전 점검 사항
    # -------------------------------------------------------------------------
    st.markdown("#### 실행 전 점검 사항")

    checks = [
        ("모델 파일", MODEL_PATH.exists()),
        ("메타데이터 파일", META_PATH.exists()),
        ("데이터 파일", DATA_PATH.exists()),
        (
            "타깃 변수 Loan_Status",
            "Loan_Status" in df.columns,
        ),
    ]

    for label, passed in checks:
        if passed:
            st.success(f"확인 완료: {label}")
        else:
            st.error(f"확인 필요: {label}")

    st.info(
        "주의: 모델 비교 결과와 테스트 성능은 제공된 데이터셋에서 "
        "산출한 교육용 결과입니다. 실제 금융 심사에 사용하기 전에는 "
        "외부 검증, 공정성 평가 및 관련 규정 검토가 필요합니다."
    )
