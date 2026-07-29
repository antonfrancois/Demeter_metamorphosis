if __name__ == "__main__":
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from .main import main

    main()
