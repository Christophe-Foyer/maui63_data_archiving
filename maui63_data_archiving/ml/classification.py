class FewShotClassifier:
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", batch_size=32, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute_embeddings(self, dataset):
        model = AutoModel.from_pretrained(self.model_name).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_name)

        model.eval()    
        all_embeds = []
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for batch in tqdm(loader, desc="Extracting DINOv3 embeddings"):
            images = [Image.fromarray(im_batch.numpy()) for im_batch in batch["image"]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

                # use global pooled representation
                embeds = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(1)

            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0)

    def find_similar_images(dataset_embeds, reference_embeds, top_k=-1, use_reference_mean=True):
        if use_reference_mean:
            # compute cosine similarity between dataset and mean of reference embeddings
            mean_ref_embeds = reference_embeds.mean(dim=0)
            sims = dataset_embeds.matmul(mean_ref_embeds)
            max_sim = sims
        else:
            # compute cosine similarity between dataset and reference embeddings
            sims = dataset_embeds.matmul(reference_embeds.T)

            # best match per dataset image
            max_sim, _ = sims.max(dim=1)

        if top_k > 0:
            top_indices = torch.topk(max_sim, top_k).indices.numpy()
        else:
            top_indices = torch.topk(max_sim, len(max_sim)).indices.numpy()
        return top_indices, max_sim.numpy()

    def train_classifier(self, dataset):
        # Assumes the dataset label key contains an actual class value
        pass
