import streamlit as st
import pandas as pd
import plotly.figure_factory as ff


# Toxicity labels
label_list = ['toxic', 'severe_toxic', 'obscene', 
				'threat', 'insult', 'identity_hate']

def metrics_page():
    # Model selectbox
    model_choice_metric = st.selectbox('Choose your model', ['BERT', 'ALBERT', 'DISTILBERT'])

    # Metric data load
    metrics_df = pd.read_csv('data/bert_metrics.csv')
    conf_matrix_df = pd.read_csv('data/bert_confusion_matrix.csv')

    # Displaying model metrics as a table
    st.write('### Model Metrics:')
    metrics_df_reset = metrics_df.reset_index(drop=True)

    # Display the table without the index column
    st.dataframe(metrics_df_reset, hide_index=True)

    st.write('### Confusion Matrices:')

    # Confusion matrix data load
    for i, row in conf_matrix_df.iterrows():

        matrix = [[row['FN'], row['TP']], [row['TN'], row['FP']]]
        columns_x = [f'Predicted not {label_list[i]}', f'Predicted {label_list[i]}']
        columns_y = [f'{label_list[i]}', f'Not {label_list[i]}']

        # Create a Plotly heatmap
        fig = ff.create_annotated_heatmap(
            z=matrix,
            x=columns_x,
            y=columns_y,
            colorscale='Blues',
            showscale=True,
            colorbar_title='Count',
            colorbar_tickprefix=' ',
        )

        # Customize layout for better visuals
        fig.update_layout(
            xaxis=dict(title='Predicted Labels'),
            yaxis=dict(title='True Labels'),
            autosize=True,
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        