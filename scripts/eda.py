import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
from deepchecks.tabular import Dataset


def save_target_distribution(train_data, save_path, target_column="satisfaction"):
    plt.figure(figsize=(5, 5))
    ax = sns.countplot(data=train_data, x = target_column, hue = target_column, palette=["lightcoral", "lightgreen"], legend=False)
    plt.title("Target Variable Distribution")
    plt.xlabel(target_column.title())
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    for p in ax.patches:
        height = p.get_height()  
        ax.text(p.get_x() + p.get_width() / 2, height + 0.1,  
                f'{int(height)}', 
                ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()

    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    
    file_to_save = save_path / 'target_variable_distribution.png'

    plt.savefig(file_to_save)

    print(f"Target variable distribution plot saved in: \033[1m{file_to_save}\033[0m")


def save_correlation_matrix(train_data, save_path):
    numeric_data = train_data.select_dtypes(include=['float'])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    file_to_save = save_path / 'correlation_matrix.png'

    plt.savefig(file_to_save)

    print(f"Correlation matrix saved in: \033[1m{file_to_save}\033[0m")


def validate_for_correlations(train_data, feature_target_threshold=0.92, feature_feature_threshold=0.9):
    dataset_to_check = Dataset(train_data, label="satisfaction", cat_features=[])

    check_feat_target_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(feature_target_threshold)
    check_feat_target_corr_result = check_feat_target_corr.run(dataset=dataset_to_check)

    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(threshold = feature_feature_threshold, n_pairs = 0)
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=dataset_to_check)

    if not check_feat_target_corr_result.passed_conditions():
        raise ValueError(f"There is at least one feature having a correlation higher or equal to {feature_target_threshold} with the target variable!")

    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("There are at least two features having a correlation higher or equal to {feature_feature_threshold}!")
    
    print("\033[1mCongratulations!\033[0m Feature-Target Correlations Passed!")
    print("\033[1mCongratulations!\033[0m Feature-Feature Correlations Passed!")


def save_numeric_feat_target_plots(train_data, save_path, target_column="satisfaction"):
    numeric_columns = ['age', 'flight_distance', 'departure_delay_in_minutes']
    n_rows = 2
    n_cols = 2
    
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[1, 1])

    for i, column in enumerate(numeric_columns[:-1]):
        ax = fig.add_subplot(gs[0, i])
        sns.kdeplot(data=train_data, x=column, hue=target_column, fill=True, ax=ax, common_norm=False)
        ax.set_title(f'Density Plot of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Density')

    ax = fig.add_subplot(gs[1, :])
    sns.kdeplot(data=train_data, x=numeric_columns[2], hue=target_column, fill=True, ax=ax, common_norm=False)
    ax.set_title(f'Density Plot of {numeric_columns[-1]}')
    ax.set_xlabel(numeric_columns[-1])
    ax.set_ylabel('Density')

    plt.tight_layout()

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    file_to_save = save_path / 'numeric_feat_target_plots.png'

    plt.savefig(file_to_save)

    print(f"Numeric features vs. Target variable plots saved in: \033[1m{file_to_save}\033[0m")


def save_cat_feat_target_plots(train_data, save_path, target_column="satisfaction"):
    numeric_cols = ['age', 'flight_distance', 'departure_delay_in_minutes']
    ordinal_cols = list(set(train_data.select_dtypes(include=['number']).columns) - set(numeric_cols))
    
    n_cols = 3
    n_rows = (len(ordinal_cols) + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, column in enumerate(ordinal_cols):
        sns.countplot(data=train_data, x=column, hue=target_column, ax=axes[i])
        axes[i].set_title(f'Count Plot of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')  

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    file_to_save = save_path / 'cat_feat_target_plots.png'

    plt.savefig(file_to_save)

    print(f"Categorical features vs. Target variable plots saved in: \033[1m{file_to_save}\033[0m")


@click.command()
@click.option('--train-data-path', type=str, help="Path to the training data set.")
@click.option('--plot-to', type=str, help="Path to directory where the plots from the eda will be saved to.")
def main(train_data_path, plot_to):
    train_data = pd.read_csv(train_data_path)
    plot_to_path = Path(plot_to)

    if not plot_to_path.exists():
        plot_to_path.mkdir(parents=True, exist_ok=True)

    save_target_distribution(train_data=train_data, save_path=plot_to_path, target_column="satisfaction")

    validate_for_correlations(train_data, feature_target_threshold=0.92, feature_feature_threshold=0.9)

    save_correlation_matrix(train_data=train_data, save_path=plot_to_path)

    save_numeric_feat_target_plots(train_data=train_data, save_path=plot_to_path, target_column="satisfaction")
    
    save_cat_feat_target_plots(train_data=train_data, save_path=plot_to_path, target_column="satisfaction")

if __name__ == '__main__':
    main()