A local end-to-end demo for predicting 30-day readmission from synthetic EHR data.
Steps:
1. pip install -r requirements.txt
2. python src/generate_data.py
3. python src/preprocess.py
4. python src/train_model.py
5. uvicorn src.api.app:app --reload
6. streamlit run dashboard/streamlit_app.py
