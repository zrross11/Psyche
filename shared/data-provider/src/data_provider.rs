use crate::{
    http::HttpDataProvider, DummyDataProvider, TokenizedDataProvider,
};


pub enum DataProvider {
    Http(HttpDataProvider),
    Dummy(DummyDataProvider),
}

impl TokenizedDataProvider for DataProvider {
    async fn get_samples(
        &mut self,
        data_ids: &[psyche_core::BatchId],
    ) -> anyhow::Result<Vec<Vec<i32>>> {
        match self {
            DataProvider::Http(data_provider_http_client) => {
                data_provider_http_client.get_samples(data_ids).await
            }
            DataProvider::Dummy(dummy_data_provider) => {
                dummy_data_provider.get_samples(data_ids).await
            }
        }
    }
}
