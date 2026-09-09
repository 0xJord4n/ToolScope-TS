export function invalidJsonData(context: string, detail: string): TypeError {
  return new TypeError(`Invalid ${context}: expected strict JSON data (${detail})`);
}

function defineEnumerable(target: Record<string, unknown>, key: string, value: unknown): void {
  Object.defineProperty(target, key, {
    value,
    enumerable: true,
    configurable: true,
    writable: true,
  });
}

function hasInheritedEnumerableState(value: object): boolean {
  let prototype = Object.getPrototypeOf(value);
  while (prototype !== null) {
    for (const key of Reflect.ownKeys(prototype)) {
      if (Object.getOwnPropertyDescriptor(prototype, key)?.enumerable) return true;
    }
    prototype = Object.getPrototypeOf(prototype);
  }
  return false;
}

export function cloneStrictJsonData(
  value: unknown,
  context: string,
  active = new Set<object>(),
): unknown {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (Number.isFinite(value)) return value;
    throw invalidJsonData(context, "numbers must be finite");
  }
  if (typeof value !== "object") {
    throw invalidJsonData(
      context,
      "only null, booleans, finite numbers, strings, arrays, and objects are allowed",
    );
  }
  if (active.has(value)) throw invalidJsonData(context, "cycles are forbidden");
  active.add(value);
  try {
    if (Array.isArray(value)) {
      if (hasInheritedEnumerableState(value)) {
        throw invalidJsonData(context, "inherited enumerable state is forbidden");
      }
      if (Object.getPrototypeOf(value) !== Array.prototype) {
        throw invalidJsonData(context, "arrays must use the standard array prototype");
      }
      if (Object.getOwnPropertySymbols(value).length > 0) {
        throw invalidJsonData(context, "symbol keys are forbidden");
      }
      const ownNames = Object.getOwnPropertyNames(value);
      if (
        ownNames.some(
          (key) =>
            key !== "length" && (!/^(?:0|[1-9]\d*)$/.test(key) || Number(key) >= value.length),
        )
      ) {
        throw invalidJsonData(context, "arrays may contain only indexed values");
      }
      const clone: unknown[] = [];
      for (let index = 0; index < value.length; index += 1) {
        if (!Object.hasOwn(value, index)) {
          throw invalidJsonData(context, "arrays must be dense");
        }
        const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
        if (!descriptor || !("value" in descriptor) || !descriptor.enumerable) {
          throw invalidJsonData(context, "array values must be own enumerable data properties");
        }
        clone.push(cloneStrictJsonData(descriptor.value, context, active));
      }
      return clone;
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw invalidJsonData(context, "objects must use a plain or null prototype");
    }
    for (const key in value) {
      if (!Object.hasOwn(value, key)) {
        throw invalidJsonData(context, "inherited enumerable state is forbidden");
      }
    }
    if (Object.getOwnPropertySymbols(value).length > 0) {
      throw invalidJsonData(context, "symbol keys are forbidden");
    }
    const clone: Record<string, unknown> = {};
    for (const key of Object.getOwnPropertyNames(value)) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (!descriptor || !("value" in descriptor) || !descriptor.enumerable) {
        throw invalidJsonData(context, "object values must be own enumerable data properties");
      }
      defineEnumerable(clone, key, cloneStrictJsonData(descriptor.value, context, active));
    }
    return clone;
  } finally {
    active.delete(value);
  }
}
