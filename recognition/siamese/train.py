from params import (MODELPATH, IMAGEPATH, EPOCHS_SIAMESE, EPOCHS_CLS,
                    TRIPLET_MARGIN, LR_SIAMESE, LR_CLS)

import tqdm



def train_siamese(encoder, loader, device):
    """train siamese encoder with triplet loss"""
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=LR_SIAMESE, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)

    for epoch in range(EPOCHS_SIAMESE):
        total_loss = 0
        for anc, pos, neg, _ in tqdm(loader, desc=f"Siamese Epoch {epoch+1}/{EPOCHS_SIAMESE}", leave=False):
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            za, zp, zn = encoder(anc), encoder(pos), encoder(neg)
            loss = criterion(za, zp, zn)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"[Siamese] epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
