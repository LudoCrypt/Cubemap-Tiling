# Cubemap-Tiling

    Stable Diffusion Cubemap Tiling

----

⚠ Not complete ⚠


The cubemap output order:

```python
'''
latent tensor:
    +---+---+---+
    | S | E | N |   S-front, E-right, N-back 
    +---+---+---+
    | B | T | W |   B-bottom, T-top, W-left
    +---+---+---+
cubemap box: (note tiles are drew on outer side of the box)
        +---+
        | T |
    +---+---+---+---+
    | W | S | E | N |
    +---+---+---+---+
        | B |
        +---+
'''
```
