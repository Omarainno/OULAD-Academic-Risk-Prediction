import pandas as pd
from src.features import build_dataset

def test_boundary_is_half():
    d = {
        "courses": pd.DataFrame({
            "code_module":["AAA"], "code_presentation":["2020J"],
            "module_presentation_length":[100]
        }),
        "studentInfo": pd.DataFrame({
            "code_module":["AAA"], "code_presentation":["2020J"],
            "id_student":[1], "final_result":["Pass"],
            "gender":["M"], "region":["R"], "highest_education":["HE"],
            "imd_band":["10-20"], "age_band":["0-35"],
            "num_of_prev_attempts":[0], "studied_credits":[60], "disability":["N"]
        }),
        "studentRegistration": pd.DataFrame({
            "code_module":["AAA"],"code_presentation":["2020J"],
            "id_student":[1],"date_registration":[-10]
        }),
        "assessments": pd.DataFrame({
            "code_module":["AAA"],"code_presentation":["2020J"],
            "id_assessment":[10],"assessment_type":["TMA"],
            "date":[40],"weight":[10]
        }),
        "studentAssessment": pd.DataFrame({
            "id_assessment":[10],"id_student":[1],"date_submitted":[39],
            "is_banked":[0],"score":[80]
        }),
        "vle": pd.DataFrame({
            "id_site":[100],"code_module":["AAA"],
            "code_presentation":["2020J"],"activity_type":["resource"]
        }),
        "studentVle": pd.DataFrame({
            "code_module":["AAA"],"code_presentation":["2020J"],
            "id_student":[1],"id_site":[100],"date":[45],"sum_click":[5]
        })
    }
    x=build_dataset(d,0.50)
    assert int(x["cutoff_day"].iloc[0]) == 50
    assert int(x["total_clicks"].iloc[0]) == 5
