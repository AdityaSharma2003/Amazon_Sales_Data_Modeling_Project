import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    df['discounted_price'] = df['discounted_price'].str[1:]
    df['actual_price'] = df['actual_price'].str[1:]

    df['discounted_price'] = pd.to_numeric(df['discounted_price'].str.replace(',', ''), errors='coerce').fillna(0).astype(int)
    df['actual_price'] = pd.to_numeric(df['actual_price'].str.replace(',', ''), errors='coerce').fillna(0).astype(int)
    df['discount_percentage'] = pd.to_numeric(df['discount_percentage'].str.replace('%', ''), errors='coerce').fillna(0).astype(int)
    df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce').fillna(0).astype(int)

    df['rating'] = pd.to_numeric(df['rating'].str.replace('|','0'),errors='coerce').fillna(0).astype(float)

    df = df.drop_duplicates().reset_index(drop=True)
    df['item_id'] = df.index

    product_description_dim = df[['product_name','about_product','img_link','product_link']].reset_index(drop=True)
    product_description_dim['product_description_id'] = product_description_dim.index
    product_description_dim = product_description_dim[['product_description_id','product_name','about_product','img_link','product_link']]

    user_description_dim = df[['user_id','user_name']].reset_index(drop=True)
    user_description_dim['user_description_id'] = user_description_dim.index
    user_description_dim = user_description_dim[['user_description_id','user_id','user_name']]

    product_review_dim = df[['review_id','review_title','review_content']].reset_index(drop=True)
    product_review_dim['product_review_id'] = product_review_dim.index
    product_review_dim = product_review_dim[['product_review_id','review_id','review_title','review_content']]

    product_category_dim = df[['category']].reset_index(drop=True)
    product_category_dim['product_category_id'] = product_category_dim.index
    product_category_dim = product_category_dim[['product_category_id','category']]

    rating_dim = df[['rating','rating_count']].reset_index(drop=True)
    rating_dim['rating_id'] = rating_dim.index
    rating_dim = rating_dim[['rating_id','rating','rating_count']]

    fact_table = df.merge(product_description_dim, left_on='item_id', right_on='product_description_id') \
             .merge(user_description_dim, left_on='item_id', right_on='user_description_id') \
             .merge(product_review_dim, left_on='item_id', right_on='product_review_id') \
             .merge(product_category_dim, left_on='item_id', right_on='product_category_id') \
             .merge(rating_dim, left_on='item_id', right_on='rating_id')\
             [['item_id','product_id', 'product_description_id', 'user_description_id',
               'product_review_id', 'product_category_id', 'rating_id', 'discounted_price', 'actual_price',
               'discount_percentage']]

    return {
        "product_description_dim":product_description_dim.to_dict(orient="dict"),
        "user_description_dim":user_description_dim.to_dict(orient="dict"),
        "product_review_dim":product_review_dim.to_dict(orient="dict"),
        "product_category_dim":product_category_dim.to_dict(orient="dict"),
        "rating_dim":rating_dim.to_dict(orient="dict"),
        "fact_table":fact_table.to_dict(orient="dict")
    }


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
