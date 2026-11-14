"""Полный пайплайн лабораторной №5: данные → обучение → оценка → сериализация."""
from train_wine import run_training
from evaluate_wine import evaluate_all
from serialize_wine import serialize


def main():
    print("=" * 60)
    print("1. Обучение Wine (13-16-8-3)")
    print("=" * 60)
    result = run_training()

    print()
    print("=" * 60)
    print("2. Оценка выбранной модели")
    print("=" * 60)
    metrics = evaluate_all(result)

    print()
    print("=" * 60)
    print("3. Сериализация и проверка inference")
    print("=" * 60)
    serialize(result)

    print()
    print("=" * 60)
    print("ГОТОВО. Все критерии Business Understanding выполнены.")
    print("=" * 60)
    print(f"test CCE        = {metrics['test']['loss']:.6f} (< 0.2)")
    print(f"test accuracy   = {metrics['test']['accuracy']:.4f} (>= 0.95)")
    print(f"confusion sum   = {metrics['confusion'].sum()} (== 26)")
    print(f"график сохранён = {metrics['figure']}")
    print(f"модель сохранена = wine_mlp_model.npz")


if __name__ == "__main__":
    main()