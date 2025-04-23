from river import metrics, utils, preprocessing, linear_model, ensemble, optim, evaluate
from models import ADWINBaggingRegressor

def eval_classifier(dataset):
    overall_metric = metrics.Accuracy()
    rolling_metric = utils.Rolling(metrics.Accuracy(), window_size=100) 
    
    model_plain = (
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression(
            optimizer=optim.Adam(lr=3e-4)
        )
    )
    
    overall_plain_result = evaluate.progressive_val_score(
        dataset, 
        (overall_plain_model := model_plain.clone()), 
        overall_metric
    )
    
    rolling_overall_result = evaluate.progressive_val_score(
        dataset, 
        model_plain.clone(), 
        rolling_metric
    )
    
    print(f"plain: {overall_plain_result} | {rolling_overall_result}")


    
    model_Ada = ensemble.AdaBoostClassifier(
        model=model_plain.clone(),
        n_models=10,
        seed=42
    )
    overall_Ada_result = evaluate.progressive_val_score(
        dataset, 
        model_Ada.clone(), 
        overall_metric
    )

    rolling_Ada_result = evaluate.progressive_val_score(
        dataset, 
        model_Ada.clone(), 
        rolling_metric
    )

    print(f"Ada: {overall_Ada_result} | {rolling_Ada_result}")
    
    
    
    model_ADWIN = ensemble.ADWINBaggingClassifier(
        model=model_plain.clone(),
        n_models=10,
        seed=42
    )
    overall_ADWIN_result = evaluate.progressive_val_score(
        dataset, 
        model_ADWIN.clone(), 
        overall_metric
    )

    rolling_ADWIN_result = evaluate.progressive_val_score(
        dataset, 
        model_ADWIN.clone(), 
        rolling_metric
    )
    
    
    print(f"ADWIN: {overall_ADWIN_result} | {rolling_ADWIN_result}")
    
def eval_regressor(dataset):
    overall_metric = metrics.MAE() 
    rolling_metric = utils.Rolling(metrics.MAE(), window_size=1000) #metrics.F1()
    
    model_plain = (
        preprocessing.OrdinalEncoder() |
        preprocessing.StandardScaler() |
        linear_model.LinearRegression(
            optimizer=optim.Adam(lr=3e-4)
        )
    )
    
    overall_plain_result = evaluate.progressive_val_score(
        dataset, 
        model_plain.clone(), 
        overall_metric
    )
    
    rolling_overall_result = evaluate.progressive_val_score(
        dataset, 
        model_plain.clone(), 
        rolling_metric
    )
    
    print(f"plain: {overall_plain_result} | {rolling_overall_result}")


    
    model_Bagging = ensemble.BaggingRegressor(
        model=model_plain.clone(),
        n_models=10,
        seed=42
    )
    overall_Bagging_result = evaluate.progressive_val_score(
        dataset, 
        model_Bagging.clone(), 
        overall_metric
    )

    rolling_Bagging_result = evaluate.progressive_val_score(
        dataset, 
        model_Bagging.clone(), 
        rolling_metric
    )

    print(f"Bagging: {overall_Bagging_result} | {rolling_Bagging_result}")
    
    
    
    model_ADWIN = ADWINBaggingRegressor(
        model=model_plain.clone(),
        n_models=10,
        seed=42
    )
    overall_ADWIN_result = evaluate.progressive_val_score(
        dataset, 
        model_ADWIN.clone(), 
        overall_metric
    )

    rolling_ADWIN_result = evaluate.progressive_val_score(
        dataset, 
        model_ADWIN.clone(), 
        rolling_metric
    )
    
    
    print(f"ADWIN: {overall_ADWIN_result} | {rolling_ADWIN_result}")