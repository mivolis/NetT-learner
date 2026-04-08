from nett_learner import generate_data, NetTLinear, NetTGCN, get_device

sim = generate_data(
    num=300,
    p_edge=0.02,
    k=10,
    seed=42,
    balance=0.5,
    y_model="gcn",
    graph="sbm",
)

linear = NetTLinear(kernel="kr_rbf").fit(sim)
linear_out = linear.estimate_effects(sim, X_cate=sim.X_raw)
print("NetT-linear:", linear_out["direct_mean"], linear_out["peer_mean"], linear_out["selected_kernel"])

gcn = NetTGCN(device=get_device(), epochs=50, hidden_features=32, kernel="kr_rbf", seed=42).fit(sim)
gcn_out = gcn.estimate_effects(sim, X_cate=sim.X_raw)
print("NetT-GCN:", gcn_out["direct_mean"], gcn_out["peer_mean"], gcn_out["selected_kernel"])
